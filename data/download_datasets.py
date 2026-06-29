"""
Dataset Download Scripts for Marathi Speech Data

Supported datasets:
  1. Mozilla Common Voice — Marathi split (crowdsourced read speech)
  2. OpenSLR Marathi — Various open Marathi speech corpora
  3. Google FLEURS — Marathi split (for evaluation)

Usage:
    python data/download_datasets.py --dataset common_voice --output_dir ./raw_data
    python data/download_datasets.py --dataset openslr --output_dir ./raw_data
"""

import os
import argparse
import tarfile
import urllib.request
from pathlib import Path


def download_file(url: str, dest_path: str):
    """Download a file with progress reporting."""
    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return

    print(f"  Downloading: {url}")
    print(f"  Saving to: {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    def _progress(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            print(f"\r  Progress: {pct}% ({mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print()


def download_common_voice(output_dir: str):
    """
    Download Mozilla Common Voice Marathi using the HuggingFace `datasets` library.
    
    This is the recommended approach as it handles authentication
    and versioning automatically.
    """
    try:
        from datasets import load_dataset

        print("Downloading Common Voice Marathi via HuggingFace datasets...")
        print("Note: You may need to accept the dataset terms at")
        print("https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1")
        print()

        cv_dir = os.path.join(output_dir, "common_voice_mr")
        os.makedirs(cv_dir, exist_ok=True)

        # Download Marathi split
        ds = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "mr",
            split="train+validation+test",
            trust_remote_code=True,
        )

        # Save as individual audio files + manifest
        manifest_rows = []
        audio_dir = os.path.join(cv_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        for i, sample in enumerate(ds):
            audio_path = os.path.join(audio_dir, f"cv_mr_{i:06d}.wav")
            # Save audio
            import soundfile as sf
            sf.write(audio_path, sample["audio"]["array"], sample["audio"]["sampling_rate"])
            manifest_rows.append({
                "audio_path": audio_path,
                "transcript": sample["sentence"],
                "duration": len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"],
            })

        # Save manifest
        import pandas as pd
        df = pd.DataFrame(manifest_rows)
        manifest_path = os.path.join(cv_dir, "manifest.csv")
        df.to_csv(manifest_path, index=False)
        print(f"Saved {len(df)} samples to {manifest_path}")

    except ImportError:
        print("ERROR: Install the `datasets` and `soundfile` packages:")
        print("  pip install datasets soundfile")
    except Exception as e:
        print(f"ERROR: {e}")
        print("You may need to authenticate with HuggingFace:")
        print("  huggingface-cli login")


def download_openslr_marathi(output_dir: str):
    """
    Download OpenSLR Marathi datasets.
    
    SLR64: Marathi speech corpus (mr_in_female)
    Contains read speech in Marathi with transcriptions.
    """
    slr_dir = os.path.join(output_dir, "openslr_mr")
    os.makedirs(slr_dir, exist_ok=True)

    # OpenSLR 64 — Marathi (mr_in_female)
    urls = [
        "https://www.openslr.org/resources/64/mr_in_female.zip",
    ]

    for url in urls:
        filename = url.split("/")[-1]
        dest = os.path.join(slr_dir, filename)
        download_file(url, dest)

        # Extract
        if filename.endswith(".zip"):
            import zipfile
            print(f"  Extracting {filename}...")
            with zipfile.ZipFile(dest, "r") as zf:
                zf.extractall(slr_dir)
            print(f"  Extracted to {slr_dir}")
        elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
            print(f"  Extracting {filename}...")
            with tarfile.open(dest, "r:gz") as tf:
                tf.extractall(slr_dir)
            print(f"  Extracted to {slr_dir}")

    print(f"\nOpenSLR Marathi data downloaded to: {slr_dir}")
    print("Run `python data/prepare_data.py` to create manifests.")


def download_fleurs_marathi(output_dir: str):
    """
    Download Google FLEURS Marathi for evaluation.
    """
    try:
        from datasets import load_dataset

        print("Downloading FLEURS Marathi...")
        fleurs_dir = os.path.join(output_dir, "fleurs_mr")
        os.makedirs(fleurs_dir, exist_ok=True)

        ds = load_dataset("google/fleurs", "mr_in", split="test", trust_remote_code=True)

        manifest_rows = []
        audio_dir = os.path.join(fleurs_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        for i, sample in enumerate(ds):
            audio_path = os.path.join(audio_dir, f"fleurs_mr_{i:04d}.wav")
            import soundfile as sf
            sf.write(audio_path, sample["audio"]["array"], sample["audio"]["sampling_rate"])
            manifest_rows.append({
                "audio_path": audio_path,
                "transcript": sample["transcription"],
                "duration": len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"],
            })

        import pandas as pd
        df = pd.DataFrame(manifest_rows)
        manifest_path = os.path.join(fleurs_dir, "manifest.csv")
        df.to_csv(manifest_path, index=False)
        print(f"Saved {len(df)} FLEURS test samples to {manifest_path}")

    except ImportError:
        print("ERROR: Install the `datasets` and `soundfile` packages.")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Marathi speech datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["common_voice", "openslr", "fleurs", "all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="raw_data",
        help="Directory to save downloaded data",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset in ("common_voice", "all"):
        print("=" * 60)
        print("Downloading Mozilla Common Voice — Marathi")
        print("=" * 60)
        download_common_voice(args.output_dir)

    if args.dataset in ("openslr", "all"):
        print("\n" + "=" * 60)
        print("Downloading OpenSLR — Marathi")
        print("=" * 60)
        download_openslr_marathi(args.output_dir)

    if args.dataset in ("fleurs", "all"):
        print("\n" + "=" * 60)
        print("Downloading Google FLEURS — Marathi")
        print("=" * 60)
        download_fleurs_marathi(args.output_dir)

    print("\nDone! Next step: python data/prepare_data.py --input_dir", args.output_dir)
