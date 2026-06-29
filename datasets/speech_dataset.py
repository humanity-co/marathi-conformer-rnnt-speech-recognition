"""
Speech Dataset and DataLoader for Marathi Conformer RNN-T

Handles loading audio, extracting/loading cached features, tokenizing text,
and creating batches with dynamic padding for efficient training.

Why Mel Spectrograms for ASR:
  • Mel spectrograms approximate human auditory perception
  • They compress raw audio into a compact 2D representation
  • The mel frequency scale emphasizes lower frequencies where
    most speech information lives
  • Log scaling compresses dynamic range, matching perception
  • 80 mel bins capture sufficient detail for phoneme discrimination
"""

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from tokenizer.charset import MarathiCharTokenizer
from augmentation.audio_augment import AudioAugmentor
from augmentation.spec_augment import SpecAugment


class FeatureExtractor:
    """
    Extract 80-dimensional log-mel spectrogram features from audio waveforms.
    
    Configuration matches standard ASR settings:
      • 25ms window (win_length=400 at 16kHz)
      • 10ms hop (hop_length=160 at 16kHz)
      • 80 mel filterbanks from 0–8000 Hz
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract log-mel spectrogram.
        
        Args:
            waveform: (1, T) or (T,) raw audio
            
        Returns:
            features: (T', n_mels) — time-first mel spectrogram
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        mel = self.mel_transform(waveform)         # (1, n_mels, T')
        log_mel = self.amplitude_to_db(mel)         # (1, n_mels, T')
        log_mel = log_mel.squeeze(0).transpose(0, 1)  # (T', n_mels)
        return log_mel


class SpeechDataset(Dataset):
    """
    Speech dataset for Marathi ASR training.
    
    Reads a CSV manifest with columns: audio_path, transcript
    Supports:
      • On-the-fly audio loading and feature extraction
      • Pre-cached features (loaded from disk)
      • Audio augmentation (speed perturbation, noise injection)
      • SpecAugment on mel spectrograms
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer: MarathiCharTokenizer | None = None,
        feature_extractor: FeatureExtractor | None = None,
        audio_augmentor: AudioAugmentor | None = None,
        spec_augment: SpecAugment | None = None,
        cache_dir: str | None = None,
        max_duration: float = 15.0,
        sample_rate: int = 16000,
        is_training: bool = True,
    ):
        self.df = pd.read_csv(manifest_path)
        self.tokenizer = tokenizer or MarathiCharTokenizer()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.audio_augmentor = audio_augmentor if is_training else None
        self.spec_augment_fn = spec_augment if is_training else None
        self.cache_dir = cache_dir
        self.sample_rate = sample_rate
        self.is_training = is_training

        # Filter by max duration if duration column exists
        if "duration" in self.df.columns:
            self.df = self.df[self.df["duration"] <= max_duration].reset_index(drop=True)

        print(f"Loaded {len(self.df)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio, resample to 16kHz, convert to mono, normalize."""
        waveform, sr = torchaudio.load(path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Normalize amplitude
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        return waveform  # (1, T)

    def _get_cached_path(self, idx: int) -> str | None:
        """Get path to cached features file."""
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, f"features_{idx}.pt")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        transcript = str(row["transcript"])

        # Try to load cached features
        cached_path = self._get_cached_path(idx)
        if cached_path and os.path.exists(cached_path) and not self.is_training:
            features = torch.load(cached_path, weights_only=True)
        else:
            # Load and preprocess audio
            waveform = self._load_audio(audio_path)

            # Apply audio augmentation (speed, noise) on raw waveform
            if self.audio_augmentor is not None:
                waveform = self.audio_augmentor(waveform)

            # Extract mel features
            features = self.feature_extractor(waveform)  # (T', n_mels)

            # Apply SpecAugment on spectrogram
            if self.spec_augment_fn is not None:
                # SpecAugment expects (..., freq, time) format
                features_spec = features.transpose(0, 1).unsqueeze(0)  # (1, n_mels, T')
                features_spec = self.spec_augment_fn(features_spec)
                features = features_spec.squeeze(0).transpose(0, 1)    # (T', n_mels)

        # Tokenize transcript
        targets = torch.tensor(self.tokenizer.encode(transcript), dtype=torch.long)

        return features, targets


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for dynamic padding within a batch.
    
    Returns:
        features_padded: (B, T_max, n_mels)
        feature_lengths: (B,)
        targets_padded: (B, U_max)
        target_lengths: (B,)
    """
    features, targets = zip(*batch)

    feature_lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)

    features_padded = torch.nn.utils.rnn.pad_sequence(
        features, batch_first=True, padding_value=0.0
    )
    targets_padded = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0
    )

    return features_padded, feature_lengths, targets_padded, target_lengths


def create_dataloader(
    manifest_path: str,
    tokenizer: MarathiCharTokenizer | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    is_training: bool = True,
    cache_dir: str | None = None,
    max_duration: float = 15.0,
    audio_augmentor: AudioAugmentor | None = None,
    spec_augment: SpecAugment | None = None,
) -> DataLoader:
    """Create a DataLoader with the SpeechDataset and collate function."""
    dataset = SpeechDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        audio_augmentor=audio_augmentor,
        spec_augment=spec_augment,
        cache_dir=cache_dir,
        max_duration=max_duration,
        is_training=is_training,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_training,
    )
