"""
Data Preparation: Clean, Normalize, and Split Marathi Speech Data

This script:
  1. Reads raw manifests from downloaded datasets
  2. Normalizes Devanagari text (Unicode normalization, remove non-Devanagari)
  3. Validates audio files (check existence, duration)
  4. Merges multiple data sources
  5. Splits into train/validation/test sets (80/10/10)
  6. Outputs clean CSV manifests

Usage:
    python data/prepare_data.py --input_dir raw_data --output_dir data/manifests
"""

import os
import re
import unicodedata
import argparse
import pandas as pd
import torchaudio
from pathlib import Path


def normalize_marathi_text(text: str) -> str:
    """
    Normalize Marathi (Devanagari) text for ASR training.
    
    Steps:
      1. Unicode NFC normalization (canonical composition)
      2. Remove non-Devanagari, non-space characters
      3. Collapse multiple spaces
      4. Strip leading/trailing whitespace
    """
    text = str(text)

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Keep only Devanagari characters (U+0900–U+097F), digits (U+0966–U+096F),
    # Devanagari extended, spaces, and basic punctuation
    text = re.sub(r"[^\u0900-\u097F\u0966-\u096F\s।,.?!-]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_audio(audio_path: str, max_duration: float = 20.0) -> tuple[bool, float]:
    """
    Validate an audio file exists and has reasonable duration.
    
    Returns:
        (is_valid, duration_seconds)
    """
    if not os.path.exists(audio_path):
        return False, 0.0

    try:
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate
        if duration < 0.5 or duration > max_duration:
            return False, duration
        return True, duration
    except Exception:
        return False, 0.0


def parse_openslr_directory(data_dir: str) -> pd.DataFrame:
    """
    Parse OpenSLR-style directory with audio files and transcription files.
    Handles the common format: TSV/TXT with <audio_id> <tab> <text>
    """
    rows = []
    transcriptions = {}

    # Find transcription files
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith((".txt", ".tsv")) or fname in ("transcription", "line_index.tsv"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split("\t")
                            if len(parts) < 2:
                                parts = line.split(" ", 1)
                            if len(parts) >= 2:
                                audio_id = parts[0].replace(".wav", "")
                                text = parts[1]
                                transcriptions[audio_id] = text
                except Exception as e:
                    print(f"Warning: Could not parse {fpath}: {e}")

    # Match audio files to transcriptions
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".wav"):
                audio_id = fname.replace(".wav", "")
                audio_path = os.path.join(root, fname)
                text = transcriptions.get(audio_id) or transcriptions.get(fname)
                if text:
                    rows.append({"audio_path": audio_path, "transcript": text})

    return pd.DataFrame(rows)


def prepare_data(input_dir: str, output_dir: str, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    Main data preparation pipeline.
    
    Collects data from all sources, normalizes, validates, and splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_data = []

    # 1. Load Common Voice manifest if available
    cv_manifest = os.path.join(input_dir, "common_voice_mr", "manifest.csv")
    if os.path.exists(cv_manifest):
        print(f"Loading Common Voice Marathi: {cv_manifest}")
        df_cv = pd.read_csv(cv_manifest)
        print(f"  Found {len(df_cv)} samples")
        all_data.append(df_cv)

    # 2. Load OpenSLR data if available
    openslr_dir = os.path.join(input_dir, "openslr_mr")
    if os.path.exists(openslr_dir):
        print(f"Parsing OpenSLR Marathi: {openslr_dir}")
        df_slr = parse_openslr_directory(openslr_dir)
        print(f"  Found {len(df_slr)} samples")
        if len(df_slr) > 0:
            all_data.append(df_slr)

    # 3. Load FLEURS manifest if available
    fleurs_manifest = os.path.join(input_dir, "fleurs_mr", "manifest.csv")
    if os.path.exists(fleurs_manifest):
        print(f"Loading FLEURS Marathi: {fleurs_manifest}")
        df_fleurs = pd.read_csv(fleurs_manifest)
        print(f"  Found {len(df_fleurs)} samples")
        all_data.append(df_fleurs)

    if not all_data:
        print("ERROR: No data found! Run download_datasets.py first.")
        return

    # Merge all sources
    df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal raw samples: {len(df)}")

    # Normalize text
    print("Normalizing Marathi text...")
    df["transcript"] = df["transcript"].apply(normalize_marathi_text)

    # Remove empty transcripts
    df = df[df["transcript"].str.len() > 0].reset_index(drop=True)
    print(f"After removing empty transcripts: {len(df)}")

    # Validate audio files
    print("Validating audio files...")
    valid_mask = []
    durations = []
    for _, row in df.iterrows():
        is_valid, dur = validate_audio(row["audio_path"])
        valid_mask.append(is_valid)
        durations.append(dur)

    df["duration"] = durations
    df = df[valid_mask].reset_index(drop=True)
    print(f"After audio validation: {len(df)}")
    print(f"Total audio duration: {df['duration'].sum() / 3600:.1f} hours")

    # Shuffle and split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    df_test = df.iloc[n_train + n_val :].reset_index(drop=True)

    # Save manifests
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"\nSplit results:")
    print(f"  Train: {len(df_train)} samples → {train_path}")
    print(f"  Val:   {len(df_val)} samples → {val_path}")
    print(f"  Test:  {len(df_test)} samples → {test_path}")
    print(f"\nData preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Marathi speech data")
    parser.add_argument("--input_dir", type=str, default="raw_data", help="Raw data directory")
    parser.add_argument("--output_dir", type=str, default="data/manifests", help="Output manifest directory")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
    args = parser.parse_args()

    prepare_data(args.input_dir, args.output_dir, args.val_ratio, args.test_ratio)
