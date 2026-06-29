"""
Feature Caching — Pre-compute and save Mel Spectrograms to disk

This dramatically speeds up training by avoiding repeated audio I/O 
and feature extraction. Especially important for laptop training where
CPU resources are limited.

Usage:
    python data/cache_features.py --manifest data/manifests/train.csv --output_dir data/cached_features
"""

import os
import argparse

import torch
import torchaudio
import pandas as pd
from tqdm import tqdm


class FeatureExtractor:
    """Extract 80-dim log-mel spectrogram (same as in speech_dataset.py)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
    ):
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=0.0,
            f_max=8000.0,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel_transform(waveform)
        log_mel = self.amplitude_to_db(mel)
        log_mel = log_mel.squeeze(0).transpose(0, 1)  # (T', n_mels)
        return log_mel


def cache_features(manifest_path: str, output_dir: str):
    """
    Pre-compute mel spectrograms for all audio in a manifest.
    Saves each as features_{idx}.pt in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(manifest_path)
    extractor = FeatureExtractor()

    print(f"Caching features for {len(df)} samples...")
    print(f"Output directory: {output_dir}")

    skipped = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Caching"):
        out_path = os.path.join(output_dir, f"features_{idx}.pt")

        if os.path.exists(out_path):
            continue

        try:
            waveform, sr = torchaudio.load(row["audio_path"])
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            # Normalize
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val

            features = extractor(waveform)
            torch.save(features, out_path)

        except Exception as e:
            print(f"\nSkipping {row['audio_path']}: {e}")
            skipped += 1

    print(f"\nDone! Cached {len(df) - skipped} features. Skipped: {skipped}")
    print(f"Disk usage: {sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir)) / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache mel spectrogram features")
    parser.add_argument("--manifest", type=str, required=True, help="Path to CSV manifest")
    parser.add_argument("--output_dir", type=str, default="data/cached_features", help="Output directory")
    args = parser.parse_args()

    cache_features(args.manifest, args.output_dir)
