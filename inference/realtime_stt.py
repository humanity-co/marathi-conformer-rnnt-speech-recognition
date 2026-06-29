"""
Real-Time Marathi Speech Transcription

Records audio from the microphone and transcribes it to Marathi text
in real time using the trained Conformer RNN-T model.

Architecture:
  • PyAudio captures microphone input in chunks
  • Audio is accumulated in a buffer (2-second sliding window)
  • Feature extraction produces log-mel spectrogram
  • Conformer encoder processes the spectrogram
  • Greedy decoder produces Marathi text
  • Output is displayed live in the terminal

Usage:
    python inference/realtime_stt.py --checkpoint checkpoints/epoch_50.pt
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torchaudio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.rnnt import ConformerRNNT
from tokenizer.charset import MarathiCharTokenizer
from datasets.speech_dataset import FeatureExtractor
from decoding.greedy import greedy_decode


class RealtimeSTT:
    """Real-time Marathi speech-to-text engine."""

    def __init__(self, checkpoint_path: str):
        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.sample_rate = 16000
        self.chunk_duration = 0.25  # 250ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.window_duration = 3.0  # Process 3-second windows
        self.window_size = int(self.sample_rate * self.window_duration)
        self.overlap_duration = 1.0  # 1-second overlap
        self.overlap_size = int(self.sample_rate * self.overlap_duration)

        # Load model
        self.tokenizer = MarathiCharTokenizer()
        self.model = ConformerRNNT(vocab_size=self.tokenizer.vocab_size)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device).eval()
        self.feature_extractor = FeatureExtractor()

        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)

        print(f"Device: {self.device}")
        print(f"Model loaded: {checkpoint_path}")
        print(f"Vocab size: {self.tokenizer.vocab_size}")

    def _transcribe_chunk(self, audio_np: np.ndarray) -> str:
        """Transcribe a chunk of audio."""
        waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)

        # Normalize
        max_val = waveform.abs().max()
        if max_val > 1e-6:
            waveform = waveform / max_val

        # Extract features
        features = self.feature_extractor(waveform)  # (T', 80)
        features = features.unsqueeze(0).to(self.device)  # (1, T', 80)
        lengths = torch.tensor([features.size(1)], dtype=torch.long).to(self.device)

        # Encode and decode
        with torch.no_grad():
            enc_out, enc_lengths = self.model.encode(features, lengths)
            text = greedy_decode(
                self.model, enc_out[0, : enc_lengths[0]], self.tokenizer
            )

        return text.strip()

    def run(self):
        """Start real-time transcription from microphone."""
        try:
            import pyaudio
        except ImportError:
            print("ERROR: PyAudio not installed. Install with:")
            print("  pip install pyaudio")
            print("  (On macOS: brew install portaudio && pip install pyaudio)")
            return

        p = pyaudio.PyAudio()

        print(f"\n{'=' * 50}")
        print("🎙️  Marathi Real-Time Speech-to-Text")
        print(f"{'=' * 50}")
        print("Speak into the microphone... (Press Ctrl+C to stop)\n")

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        try:
            while True:
                # Read audio chunk
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.float32)
                self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

                # Process when we have enough audio
                if len(self.audio_buffer) >= self.window_size:
                    # Check for speech (simple energy-based VAD)
                    energy = np.mean(self.audio_buffer[-self.window_size:] ** 2)

                    if energy > 1e-6:  # Speech detected
                        text = self._transcribe_chunk(
                            self.audio_buffer[-self.window_size:]
                        )
                        if text:
                            # Clear line and print
                            sys.stdout.write(f"\r\033[K\033[1;36m[मराठी] \033[0;32m{text}\033[0m")
                            sys.stdout.flush()

                    # Keep overlap for context continuity
                    self.audio_buffer = self.audio_buffer[-self.overlap_size:]

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\nStopping...")

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Engine stopped.")


def transcribe_file(checkpoint_path: str, audio_path: str) -> str:
    """Transcribe a single audio file."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    tokenizer = MarathiCharTokenizer()
    model = ConformerRNNT(vocab_size=tokenizer.vocab_size)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    feature_extractor = FeatureExtractor()
    features = feature_extractor(waveform).unsqueeze(0).to(device)
    lengths = torch.tensor([features.size(1)], dtype=torch.long).to(device)

    with torch.no_grad():
        enc_out, enc_lengths = model.encode(features, lengths)
        text = greedy_decode(model, enc_out[0, : enc_lengths[0]], tokenizer)

    return text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marathi Real-Time STT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--file", type=str, default=None, help="Transcribe a file instead of mic")
    args = parser.parse_args()

    if args.file:
        result = transcribe_file(args.checkpoint, args.file)
        print(f"Transcription: {result}")
    else:
        engine = RealtimeSTT(args.checkpoint)
        engine.run()
