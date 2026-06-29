"""
RNN-Transducer (RNN-T) — Prediction Network, Joint Network, and Full Model

The RNN-T framework has three components:
  1. Encoder (Conformer) — processes acoustic features
  2. Prediction Network — autoregressive language model over output tokens
  3. Joint Network — combines encoder and predictor states to produce
     output distribution at every (time, label) position

Advantage over CTC:
  • RNN-T models label dependencies (previous token conditions next token)
  • Enables streaming/online decoding (output tokens as audio arrives)
  • No independence assumption between output tokens
  • Better for morphologically rich languages like Marathi

Reference: Graves, "Sequence Transduction with Recurrent Neural Networks", 2012.
"""

import torch
import torch.nn as nn

from .conformer import ConformerEncoder


class PredictionNetwork(nn.Module):
    """
    RNN Prediction Network (acts as an internal language model).
    
    Takes the sequence of previously emitted tokens and produces
    a hidden representation used by the Joint Network.
    
    Architecture: Embedding → Multi-layer LSTM
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(
        self,
        targets: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            targets: (B, U) — token indices (previous outputs)
            hidden: Optional LSTM hidden state tuple

        Returns:
            out: (B, U, hidden_size)
            hidden: Updated LSTM hidden state
        """
        embedded = self.embedding(targets)  # (B, U, embed_dim)
        out, hidden = self.lstm(embedded, hidden)
        return out, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize zero hidden state for LSTM."""
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
        )


class JointNetwork(nn.Module):
    """
    Joint Network combining Encoder and Prediction outputs.
    
    Computes: output = Linear(ReLU(Linear(enc) + Linear(pred)))
    
    The joint network produces a distribution over the vocabulary
    (including blank) at every (time_step, label_position) pair.
    """
    def __init__(
        self,
        encoder_dim: int = 256,
        prediction_dim: int = 512,
        joint_dim: int = 512,
        vocab_size: int = 100,
    ):
        super().__init__()
        self.linear_enc = nn.Linear(encoder_dim, joint_dim)
        self.linear_pred = nn.Linear(prediction_dim, joint_dim)
        self.activation = nn.ReLU()
        self.output_linear = nn.Linear(joint_dim, vocab_size)

    def forward(
        self, encoder_out: torch.Tensor, predictor_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            encoder_out: (B, T, encoder_dim)
            predictor_out: (B, U, prediction_dim)

        Returns:
            logits: (B, T, U, vocab_size)
        
        Broadcasting creates the 4D tensor:
            enc is expanded along U dimension
            pred is expanded along T dimension
        """
        # Project and broadcast: (B, T, 1, joint_dim) + (B, 1, U, joint_dim)
        enc = self.linear_enc(encoder_out).unsqueeze(2)   # (B, T, 1, J)
        pred = self.linear_pred(predictor_out).unsqueeze(1)  # (B, 1, U, J)

        joint = self.activation(enc + pred)  # (B, T, U, J)
        logits = self.output_linear(joint)   # (B, T, U, vocab_size)
        return logits


class ConformerRNNT(nn.Module):
    """
    Full Conformer RNN-T Model for Marathi Speech Recognition.
    
    Pipeline:
        Audio mel features → ConformerEncoder → encoder_out (B, T', D)
        Previous tokens → PredictionNetwork → pred_out (B, U+1, H)
        (encoder_out, pred_out) → JointNetwork → logits (B, T', U+1, V)
    
    During training, logits are passed to torchaudio.transforms.RNNTLoss.
    During inference, greedy or beam search decoding is used.
    """
    def __init__(
        self,
        vocab_size: int = 100,
        input_dim: int = 80,
        encoder_dim: int = 256,
        encoder_blocks: int = 6,
        encoder_heads: int = 4,
        conv_kernel_size: int = 15,
        pred_embed_dim: int = 256,
        pred_hidden_dim: int = 512,
        pred_layers: int = 2,
        joint_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            d_model=encoder_dim,
            num_blocks=encoder_blocks,
            num_heads=encoder_heads,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.predictor = PredictionNetwork(
            vocab_size=vocab_size,
            embed_dim=pred_embed_dim,
            hidden_size=pred_hidden_dim,
            num_layers=pred_layers,
            dropout=dropout,
        )

        self.joint = JointNetwork(
            encoder_dim=encoder_dim,
            prediction_dim=pred_hidden_dim,
            joint_dim=joint_dim,
            vocab_size=vocab_size,
        )

        self.vocab_size = vocab_size
        self.blank_id = 0  # <blank> must be at index 0 for RNN-T

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            features: (B, T, n_mels) — log-mel spectrogram
            feature_lengths: (B,) — valid lengths
            targets: (B, U) — target token sequences (without blank prefix)
            target_lengths: (B,) — target lengths

        Returns:
            logits: (B, T', U+1, vocab_size) — for RNNT loss
            enc_lengths: (B,) — encoder output lengths after subsampling
        """
        # Encode audio
        encoder_out, enc_lengths = self.encoder(features, feature_lengths)

        # Prepend blank token to targets for prediction network input
        # Prediction network sees: [<blank>, t1, t2, ..., tU]
        blank_pad = torch.zeros(
            (targets.size(0), 1), dtype=torch.long, device=targets.device
        )
        pred_input = torch.cat([blank_pad, targets], dim=1)  # (B, U+1)

        # Run prediction network
        pred_out, _ = self.predictor(pred_input)  # (B, U+1, pred_dim)

        # Joint network produces 4D logit tensor
        logits = self.joint(encoder_out, pred_out)  # (B, T', U+1, vocab)

        return logits, enc_lengths

    def encode(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder only (used during inference)."""
        return self.encoder(features, feature_lengths)
