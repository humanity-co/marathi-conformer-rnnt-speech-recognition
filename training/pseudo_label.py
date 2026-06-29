"""
Pseudo-Label Self-Training Pipeline for Semi-Supervised ASR

How pseudo-label self-training works:
  1. Train a "bootstrap" model on a small labeled dataset (20-50 hours)
  2. Collect unlabeled Marathi audio (podcasts, audiobooks, news, etc.)
  3. Use the bootstrap model to auto-transcribe unlabeled audio
  4. Filter out low-confidence transcripts (keep only high-quality pseudo labels)
  5. Combine labeled + pseudo-labeled data
  6. Retrain on the expanded dataset

Why this dramatically improves performance:
  • Effectively multiplies training data by 5-10x
  • The model learns from diverse acoustic conditions in unlabeled data
  • Iterative self-training progressively improves pseudo-label quality
  • Critical for low-resource languages like Marathi where labeled data is scarce
  
Usage:
    python training/pseudo_label.py \\
        --checkpoint checkpoints/epoch_50.pt \\
        --unlabeled_dir /path/to/unlabeled_audio \\
        --output_manifest pseudo_labels.csv \\
        --confidence_threshold 0.7
"""

import os
import sys
import argparse
import math

import torch
import torchaudio
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.rnnt import ConformerRNNT
from tokenizer.charset import MarathiCharTokenizer
from datasets.speech_dataset import FeatureExtractor
import torch.nn.functional as F


def compute_confidence(
    model: ConformerRNNT,
    encoder_out: torch.Tensor,
    tokenizer: MarathiCharTokenizer,
    max_symbols_per_step: int = 5,
) -> tuple[str, float]:
    """
    Greedy decode with confidence scoring.
    
    Confidence is computed as the average max softmax probability
    across all non-blank token emissions. Higher = more confident.
    
    Args:
        model: Trained ConformerRNNT model (eval mode)
        encoder_out: (T, D) single utterance encoder output
        tokenizer: Character tokenizer
        
    Returns:
        (decoded_text, confidence_score)
    """
    device = encoder_out.device
    T = encoder_out.size(0)

    hidden = model.predictor.init_hidden(1, device)
    last_token = torch.tensor([[model.blank_id]], dtype=torch.long, device=device)

    output_tokens = []
    token_confidences = []

    for t in range(T):
        enc_t = encoder_out[t].unsqueeze(0).unsqueeze(0)
        symbols = 0

        while symbols < max_symbols_per_step:
            pred_out, next_hidden = model.predictor(last_token, hidden)
            logits = model.joint(enc_t, pred_out)
            probs = F.softmax(logits.squeeze(), dim=-1)
            max_prob, token_id = probs.max(dim=-1)

            token_id = token_id.item()
            confidence = max_prob.item()

            if token_id == model.blank_id:
                break
            else:
                output_tokens.append(token_id)
                token_confidences.append(confidence)
                last_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                hidden = next_hidden
                symbols += 1

    text = tokenizer.decode(output_tokens)
    avg_confidence = sum(token_confidences) / max(1, len(token_confidences))

    return text, avg_confidence


@torch.no_grad()
def generate_pseudo_labels(
    checkpoint_path: str,
    unlabeled_dir: str,
    output_manifest: str,
    confidence_threshold: float = 0.7,
    max_duration: float = 20.0,
):
    """
    Generate pseudo labels for unlabeled audio using a trained model.
    
    Scans the unlabeled directory for audio files, transcribes each,
    scores confidence, and saves high-confidence pseudo labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    # Load model
    tokenizer = MarathiCharTokenizer()
    model = ConformerRNNT(vocab_size=tokenizer.vocab_size)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device).eval()
    extractor = FeatureExtractor()

    # Find audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = []
    for root, _, files in os.walk(unlabeled_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in audio_extensions:
                audio_files.append(os.path.join(root, f))

    print(f"Found {len(audio_files)} audio files in {unlabeled_dir}")
    print(f"Confidence threshold: {confidence_threshold}")

    results = []
    kept = 0
    rejected = 0

    for audio_path in tqdm(audio_files, desc="Generating pseudo labels"):
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            # Check duration
            duration = waveform.shape[1] / 16000
            if duration < 0.5 or duration > max_duration:
                continue

            # Normalize
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val

            # Extract features
            features = extractor(waveform).unsqueeze(0).to(device)  # (1, T, 80)
            lengths = torch.tensor([features.size(1)], dtype=torch.long).to(device)

            # Encode
            enc_out, enc_lengths = model.encode(features, lengths)

            # Decode with confidence
            text, confidence = compute_confidence(
                model, enc_out[0, : enc_lengths[0]], tokenizer
            )

            if confidence >= confidence_threshold and len(text.strip()) > 0:
                results.append({
                    "audio_path": audio_path,
                    "transcript": text.strip(),
                    "duration": duration,
                    "confidence": round(confidence, 4),
                })
                kept += 1
            else:
                rejected += 1

        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            continue

    # Save results
    if results:
        os.makedirs(os.path.dirname(output_manifest) or ".", exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(output_manifest, index=False)
        print(f"\n{'=' * 50}")
        print(f"Pseudo-label generation complete!")
        print(f"  Kept: {kept} (avg confidence: {df['confidence'].mean():.3f})")
        print(f"  Rejected: {rejected}")
        print(f"  Saved to: {output_manifest}")
        print(f"{'=' * 50}")
    else:
        print("No pseudo labels generated. Try lowering the confidence threshold.")


def merge_manifests(labeled_manifest: str, pseudo_manifest: str, output_path: str):
    """
    Merge labeled and pseudo-labeled manifests for retraining.
    
    Args:
        labeled_manifest: Path to original labeled manifest CSV
        pseudo_manifest: Path to pseudo-label manifest CSV
        output_path: Path for merged output CSV
    """
    df_labeled = pd.read_csv(labeled_manifest)
    df_pseudo = pd.read_csv(pseudo_manifest)

    # Drop confidence column from pseudo labels (not needed for training)
    if "confidence" in df_pseudo.columns:
        df_pseudo = df_pseudo.drop(columns=["confidence"])

    df_merged = pd.concat([df_labeled, df_pseudo], ignore_index=True)
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    df_merged.to_csv(output_path, index=False)

    print(f"Merged manifest: {len(df_labeled)} labeled + {len(df_pseudo)} pseudo = {len(df_merged)} total")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-label self-training")
    sub = parser.add_subparsers(dest="command")

    # Generate pseudo labels
    gen = sub.add_parser("generate", help="Generate pseudo labels")
    gen.add_argument("--checkpoint", type=str, required=True)
    gen.add_argument("--unlabeled_dir", type=str, required=True)
    gen.add_argument("--output_manifest", type=str, default="pseudo_labels.csv")
    gen.add_argument("--confidence_threshold", type=float, default=0.7)
    gen.add_argument("--max_duration", type=float, default=20.0)

    # Merge manifests
    merge = sub.add_parser("merge", help="Merge labeled + pseudo-labeled manifests")
    merge.add_argument("--labeled", type=str, required=True)
    merge.add_argument("--pseudo", type=str, required=True)
    merge.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    if args.command == "generate":
        generate_pseudo_labels(
            args.checkpoint,
            args.unlabeled_dir,
            args.output_manifest,
            args.confidence_threshold,
            args.max_duration,
        )
    elif args.command == "merge":
        merge_manifests(args.labeled, args.pseudo, args.output)
    else:
        parser.print_help()
