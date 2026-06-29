"""
Full Training Pipeline for Marathi Conformer RNN-T

Features:
  • Device auto-detection: CUDA → MPS (Apple Silicon) → CPU
  • RNN-T loss via torchaudio.transforms.RNNTLoss
  • AdamW optimizer with warmup + cosine annealing LR schedule
  • Gradient clipping and gradient accumulation
  • Validation loop with greedy WER/CER computation
  • Checkpoint saving/resuming
  • Console logging of loss, WER, CER

Usage:
    # Train from scratch
    python training/train.py --config configs/default.yaml

    # Resume training
    python training/train.py --config configs/default.yaml --resume checkpoints/epoch_10.pt

    # Smoke test (5 steps on dummy data)
    python training/train.py --config configs/default.yaml --max_steps 5 --batch_size 2
"""

import os
import sys
import math
import argparse
import time

import yaml
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.rnnt import ConformerRNNT
from tokenizer.charset import MarathiCharTokenizer
from datasets.speech_dataset import SpeechDataset, collate_fn, FeatureExtractor
from augmentation.spec_augment import SpecAugment
from augmentation.audio_augment import AudioAugmentor
from decoding.greedy import greedy_decode


def get_device() -> torch.device:
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Warmup + cosine annealing learning rate schedule.
    
    Linear warmup for `warmup_steps`, then cosine decay to near-zero.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_wer_cer(predictions: list[str], references: list[str]) -> tuple[float, float]:
    """Compute Word Error Rate and Character Error Rate."""
    try:
        from jiwer import wer, cer
        if not predictions or not references:
            return 1.0, 1.0
        # Filter out empty strings
        pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
        if not pairs:
            return 1.0, 1.0
        preds, refs = zip(*pairs)
        return wer(list(refs), list(preds)), cer(list(refs), list(preds))
    except ImportError:
        print("Warning: jiwer not installed. Skipping WER/CER computation.")
        return -1.0, -1.0


def validate(
    model: ConformerRNNT,
    val_loader: DataLoader,
    criterion,
    tokenizer: MarathiCharTokenizer,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Run validation: compute loss + greedy decode WER/CER.
    
    Returns:
        (val_loss, val_wer, val_cer)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for features, feat_lengths, targets, target_lengths in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            feat_lengths = feat_lengths.to(device)
            target_lengths = target_lengths.to(device)

            logits, enc_lengths = model(features, feat_lengths, targets, target_lengths)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            loss = criterion(
                log_probs.cpu(),
                targets.int().cpu(),
                enc_lengths.int().cpu(),
                target_lengths.int().cpu(),
            )
            total_loss += loss.item()
            num_batches += 1

            # Greedy decode for WER/CER (first few samples)
            if len(all_preds) < 100:
                enc_out, enc_lens = model.encode(features, feat_lengths)
                for i in range(min(features.size(0), 10)):
                    pred_text = greedy_decode(model, enc_out[i, :enc_lens[i]], tokenizer)
                    ref_ids = targets[i, :target_lengths[i]].tolist()
                    ref_text = tokenizer.decode(ref_ids)
                    all_preds.append(pred_text)
                    all_refs.append(ref_text)

    avg_loss = total_loss / max(1, num_batches)
    val_wer, val_cer = compute_wer_cer(all_preds, all_refs)
    model.train()
    return avg_loss, val_wer, val_cer


def create_dummy_manifest(output_path: str, num_samples: int = 20):
    """Create a dummy manifest for smoke testing with synthetic data."""
    import pandas as pd
    import numpy as np

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio_dir = os.path.join(os.path.dirname(output_path), "dummy_audio")
    os.makedirs(audio_dir, exist_ok=True)

    rows = []
    marathi_words = ["नमस्कार", "कसे", "आहात", "धन्यवाद", "शुभ", "दिवस", "मराठी", "भाषा"]

    for i in range(num_samples):
        # Generate synthetic audio (1-3 seconds of noise)
        duration = 1.0 + np.random.random() * 2.0
        num_samples_audio = int(16000 * duration)
        waveform = torch.randn(1, num_samples_audio) * 0.1

        audio_path = os.path.join(audio_dir, f"dummy_{i:04d}.wav")
        torchaudio.save(audio_path, waveform, 16000)

        # Random Marathi transcript
        n_words = np.random.randint(1, 4)
        transcript = " ".join(np.random.choice(marathi_words, n_words))

        rows.append({
            "audio_path": audio_path,
            "transcript": transcript,
            "duration": duration,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Created dummy manifest: {output_path} ({num_samples} samples)")
    return output_path


def train(args):
    """Main training function."""
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"{'=' * 60}")
    print(f"Marathi Conformer RNN-T Training")
    print(f"{'=' * 60}")
    print(f"Device: {device}")

    # Initialize tokenizer
    tokenizer = MarathiCharTokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # Config overrides from CLI
    batch_size = args.batch_size or config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    grad_clip = config["training"]["grad_clip_norm"]
    accum_steps = config["training"]["grad_accumulation_steps"]
    warmup_steps = config["training"]["warmup_steps"]
    save_every = config["training"]["save_every_n_epochs"]
    val_every = config["training"]["val_every_n_epochs"]
    log_every = config["training"]["log_every_n_steps"]
    max_duration = config["training"]["max_duration"]

    # Resolve manifest paths
    train_manifest = config["paths"]["train_manifest"]
    val_manifest = config["paths"]["val_manifest"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    cache_dir = config["paths"].get("cache_dir")

    # If max_steps is set (smoke test mode), create dummy data
    if args.max_steps:
        print(f"\n⚡ SMOKE TEST MODE — {args.max_steps} steps")
        dummy_manifest = create_dummy_manifest("data/manifests/dummy_train.csv")
        train_manifest = dummy_manifest
        val_manifest = dummy_manifest
        epochs = 1
        save_every = 999
        val_every = 999

    # Check if manifest exists
    if not os.path.exists(train_manifest):
        print(f"\nManifest not found: {train_manifest}")
        print("Creating dummy data for smoke testing...")
        train_manifest = create_dummy_manifest("data/manifests/dummy_train.csv")
        val_manifest = train_manifest

    # Augmentation setup
    audio_aug = None
    spec_aug = None
    aug_config = config.get("augmentation", {})

    if aug_config.get("speed_perturb", {}).get("enabled", False):
        audio_aug = AudioAugmentor(
            speed_rates=aug_config["speed_perturb"].get("rates", [0.9, 1.0, 1.1]),
            speed_enabled=True,
            noise_enabled=aug_config.get("noise_injection", {}).get("enabled", False),
            noise_snr_db=aug_config.get("noise_injection", {}).get("snr_db", 20.0),
        )

    if aug_config.get("spec_augment", {}).get("enabled", False):
        sa_config = aug_config["spec_augment"]
        spec_aug = SpecAugment(
            freq_mask_param=sa_config.get("freq_mask_param", 27),
            time_mask_param=sa_config.get("time_mask_param", 100),
            num_freq_masks=sa_config.get("num_freq_masks", 2),
            num_time_masks=sa_config.get("num_time_masks", 2),
        )

    # Create datasets
    train_dataset = SpeechDataset(
        manifest_path=train_manifest,
        tokenizer=tokenizer,
        audio_augmentor=audio_aug,
        spec_augment=spec_aug,
        cache_dir=cache_dir,
        max_duration=max_duration,
        is_training=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(config["training"]["num_workers"], 2),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if os.path.exists(val_manifest):
        val_dataset = SpeechDataset(
            manifest_path=val_manifest,
            tokenizer=tokenizer,
            max_duration=max_duration,
            is_training=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    # Initialize model
    enc_config = config["encoder"]
    pred_config = config["predictor"]

    model = ConformerRNNT(
        vocab_size=vocab_size,
        input_dim=enc_config["input_dim"],
        encoder_dim=enc_config["d_model"],
        encoder_blocks=enc_config["num_blocks"],
        encoder_heads=enc_config["num_heads"],
        conv_kernel_size=enc_config["conv_kernel_size"],
        pred_embed_dim=pred_config["embed_dim"],
        pred_hidden_dim=pred_config["hidden_size"],
        pred_layers=pred_config["num_layers"],
        joint_dim=config["joint"]["joint_dim"],
        dropout=enc_config["dropout"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model parameters: {num_params:.1f}M total, {trainable_params:.1f}M trainable")

    # Loss, optimizer, scheduler
    criterion = torchaudio.transforms.RNNTLoss(blank=0, reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config["training"]["weight_decay"]
    )

    total_steps = len(train_loader) * epochs // accum_steps
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    # Checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume
    start_epoch = 1
    global_step = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint.get("epoch", 0) + 1
                global_step = checkpoint.get("global_step", 0)
            else:
                model.load_state_dict(checkpoint)
                import re
                match = re.search(r"epoch_(\d+)", args.resume)
                if match:
                    start_epoch = int(match.group(1)) + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Warning: checkpoint {args.resume} not found, starting fresh.")

    # ──────────────────────────────────────────────────────────────
    # Training Loop
    # ──────────────────────────────────────────────────────────────
    print(f"\nStarting training: {epochs} epochs, batch_size={batch_size}, "
          f"accum_steps={accum_steps}, lr={lr}")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad()

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for batch_idx, (features, feat_lengths, targets, target_lengths) in enumerate(progress):
            features = features.to(device)
            targets = targets.to(device)
            feat_lengths = feat_lengths.to(device)
            target_lengths = target_lengths.to(device)

            # Forward pass
            logits, enc_lengths = model(features, feat_lengths, targets, target_lengths)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # RNN-T loss (computed on CPU for stability/MPS compatibility)
            loss = criterion(
                log_probs.cpu(),
                targets.int().cpu(),
                enc_lengths.int().cpu(),
                target_lengths.int().cpu(),
            )
            loss = loss / accum_steps

            # Backward
            loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            step_loss = loss.item() * accum_steps
            epoch_loss += step_loss
            progress.set_postfix({
                "loss": f"{step_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # Logging
            if global_step > 0 and global_step % log_every == 0:
                avg = epoch_loss / (batch_idx + 1)
                print(f"\n  [Step {global_step}] Avg Loss: {avg:.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

            # Smoke test early exit
            if args.max_steps and (batch_idx + 1) >= args.max_steps:
                break

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(1, batch_idx + 1)
        elapsed = time.time() - epoch_start
        print(f"\nEpoch {epoch} complete — Avg Loss: {avg_epoch_loss:.4f}, "
              f"Time: {elapsed:.0f}s")

        # Validation
        if val_loader and epoch % val_every == 0:
            val_loss, val_wer, val_cer = validate(
                model, val_loader, criterion, tokenizer, device
            )
            print(f"  Validation — Loss: {val_loss:.4f}, WER: {val_wer:.2%}, CER: {val_cer:.2%}")

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_epoch_loss,
                "config": config,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        if args.max_steps:
            print(f"\n✅ Smoke test passed! Loss: {avg_epoch_loss:.4f}")
            break

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Marathi Conformer RNN-T")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps for smoke testing")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch size")
    args = parser.parse_args()

    train(args)
