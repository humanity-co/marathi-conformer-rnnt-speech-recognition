"""
Audio-Level Data Augmentation for Speech Recognition

These augmentations are applied to raw waveforms BEFORE feature extraction.
They simulate real-world audio variations to improve model robustness.

Augmentation methods:
  • Speed perturbation: Changes speaking rate without changing pitch
  • Noise injection: Adds Gaussian noise at a specified SNR
  
Why augmentation improves robustness:
  • Speed perturbation exposes the model to varied speaking rates,
    helping it generalize across fast and slow speakers
  • Noise injection trains the model to recognize speech in noisy
    environments, improving real-world performance
  • Together, these augmentations effectively multiply the training 
    data diversity, which is critical for low-resource languages
"""

import random
import torch
import torchaudio


class SpeedPerturb:
    """
    Speed perturbation by resampling the audio at different rates.

    A rate of 0.9 slows down (longer audio), 1.1 speeds up (shorter audio).
    The pitch changes slightly, which is acceptable for ASR training.

    Args:
        sample_rate: Original sample rate
        rates: List of speed factors to randomly choose from
    """

    def __init__(self, sample_rate: int = 16000, rates: list[float] | None = None):
        self.sample_rate = sample_rate
        self.rates = rates or [0.9, 1.0, 1.1]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (1, T) or (T,) audio tensor

        Returns:
            Speed-perturbed waveform at the original sample rate
        """
        rate = random.choice(self.rates)
        if rate == 1.0:
            return waveform

        # Resample: orig_freq -> new_freq effectively changes speed
        # To speed up by 1.1x: resample from sr*1.1 to sr (shorter output)
        resampler = torchaudio.transforms.Resample(
            orig_freq=int(self.sample_rate * rate),
            new_freq=self.sample_rate,
        )
        return resampler(waveform)


class NoiseInjection:
    """
    Add Gaussian noise at a specified Signal-to-Noise Ratio (SNR).

    Higher SNR = cleaner signal (less noise).
    Lower SNR = noisier signal (more noise).

    Args:
        snr_db: Signal-to-noise ratio in decibels
        p: Probability of applying noise
    """

    def __init__(self, snr_db: float = 20.0, p: float = 0.5):
        self.snr_db = snr_db
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (1, T) or (T,) audio tensor

        Returns:
            Noisy waveform
        """
        if random.random() > self.p:
            return waveform

        # Calculate signal power and desired noise power
        signal_power = waveform.pow(2).mean()
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate and add noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise


class AudioAugmentor:
    """
    Combined audio augmentation pipeline.

    Applies speed perturbation and noise injection sequentially.
    Only active during training.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        speed_rates: list[float] | None = None,
        speed_enabled: bool = True,
        noise_snr_db: float = 20.0,
        noise_enabled: bool = False,
    ):
        self.speed_perturb = SpeedPerturb(sample_rate, speed_rates) if speed_enabled else None
        self.noise_inject = NoiseInjection(noise_snr_db) if noise_enabled else None

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.speed_perturb is not None:
            waveform = self.speed_perturb(waveform)
        if self.noise_inject is not None:
            waveform = self.noise_inject(waveform)
        return waveform
