"""
SpecAugment — Spectrogram Augmentation for Speech Recognition

Applies frequency masking and time masking to log-mel spectrograms
during training to improve model robustness.

How it works:
  • Frequency masking: Zeroes out random contiguous bands along the
    frequency axis, forcing the model to not rely on specific frequency bins.
  • Time masking: Zeroes out random contiguous segments along the time
    axis, simulating missing or corrupted audio segments.

Why it helps:
  • Acts as a regularizer, reducing overfitting on small datasets
  • Makes the model invariant to minor spectral distortions
  • Simulates real-world audio degradation cheaply

Reference: Park et al., "SpecAugment: A Simple Data Augmentation
Method for Automatic Speech Recognition", 2019.
"""

import random
import torch
import torchaudio.transforms as T


class SpecAugment:
    """
    SpecAugment with frequency and time masking.

    Args:
        freq_mask_param: Max width of frequency mask (F)
        time_mask_param: Max width of time mask (T)
        num_freq_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply
        p: Probability of applying augmentation

    Usage:
        aug = SpecAugment()
        augmented_spec = aug(mel_spectrogram)
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        p: float = 1.0,
    ):
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to a spectrogram.

        Args:
            spec: Tensor of shape (..., freq, time)

        Returns:
            Augmented spectrogram of the same shape
        """
        if random.random() > self.p:
            return spec

        augmented = spec.clone()

        # Frequency masking
        for _ in range(self.num_freq_masks):
            augmented = self.freq_masking(augmented)

        # Time masking
        for _ in range(self.num_time_masks):
            augmented = self.time_masking(augmented)

        return augmented
