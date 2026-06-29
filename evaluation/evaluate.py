"""
Evaluation Script — Compute WER and CER on Test Data

Usage:
    python evaluation/evaluate.py \\
        --checkpoint checkpoints/epoch_50.pt \\
        --manifest data/manifests/test.csv \\
        --decode_method greedy \\
        --sample_size 100
"""

import os
import sys
import argparse

import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.rnnt import ConformerRNNT
from tokenizer.charset import MarathiCharTokenizer
from datasets.speech_dataset import FeatureExtractor, SpeechDataset, collate_fn
from decoding.greedy import greedy_decode
from decoding.beam_search import BeamSearchDecoder, KenLMScorer


def evaluate(args):
    """Run evaluation on a test manifest."""
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = MarathiCharTokenizer()

    # Load model
    model = ConformerRNNT(vocab_size=tokenizer.vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()
    print(f"Loaded model: {args.checkpoint}")

    # Decoder
    if args.decode_method == "beam":
        lm_scorer = KenLMScorer(args.lm_path) if args.lm_path else None
        decoder = BeamSearchDecoder(
            model, tokenizer, beam_size=args.beam_size, lm_scorer=lm_scorer
        )
    else:
        decoder = None  # Use greedy

    # Load test data
    df = pd.read_csv(args.manifest)
    if args.sample_size and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=42)
    print(f"Evaluating on {len(df)} samples")

    feature_extractor = FeatureExtractor()
    predictions = []
    references = []
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            import torchaudio
            waveform, sr = torchaudio.load(row["audio_path"])
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val

            features = feature_extractor(waveform).unsqueeze(0).to(device)
            lengths = torch.tensor([features.size(1)], dtype=torch.long).to(device)

            with torch.no_grad():
                enc_out, enc_lengths = model.encode(features, lengths)

                if decoder:
                    pred_text = decoder.decode(enc_out[0, : enc_lengths[0]])
                else:
                    pred_text = greedy_decode(
                        model, enc_out[0, : enc_lengths[0]], tokenizer
                    )

            ref_text = str(row["transcript"]).strip()
            predictions.append(pred_text)
            references.append(ref_text)

            # Print first 10 examples
            if len(predictions) <= 10:
                print(f"\n  Example {len(predictions)}:")
                print(f"    Reference:  '{ref_text}'")
                print(f"    Prediction: '{pred_text}'")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\nError: {e}")

    # Compute metrics
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results")
    print(f"{'=' * 60}")

    try:
        from jiwer import wer, cer

        # Filter out empty references
        pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
        if pairs:
            preds, refs = zip(*pairs)
            final_wer = wer(list(refs), list(preds))
            final_cer = cer(list(refs), list(preds))
            print(f"  Samples evaluated: {len(pairs)}")
            print(f"  Word Error Rate (WER):      {final_wer:.2%}")
            print(f"  Character Error Rate (CER): {final_cer:.2%}")
        else:
            print("  No valid reference/prediction pairs found.")

    except ImportError:
        print("  WARNING: jiwer not installed. Install with: pip install jiwer")

    if errors > 0:
        print(f"  Errors: {errors}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Marathi STT Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--manifest", type=str, required=True, help="Test manifest CSV")
    parser.add_argument("--decode_method", type=str, default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam_size", type=int, default=10, help="Beam size for beam search")
    parser.add_argument("--lm_path", type=str, default=None, help="KenLM model path")
    parser.add_argument("--sample_size", type=int, default=None, help="Limit evaluation samples")
    args = parser.parse_args()

    evaluate(args)
