"""
Greedy Decoder for RNN-Transducer

The simplest decoding strategy: at each encoder time step,
greedily emit the most likely non-blank token until blank is
predicted, then advance to the next time step.

Fast and suitable for real-time inference.
"""

import torch
import torch.nn.functional as F

from tokenizer.charset import MarathiCharTokenizer


@torch.no_grad()
def greedy_decode(
    model,
    encoder_out: torch.Tensor,
    tokenizer: MarathiCharTokenizer,
    max_symbols_per_step: int = 5,
) -> str:
    """
    Greedy decode a single utterance.
    
    Algorithm:
      For each encoder time step t:
        1. Feed last predicted token into prediction network
        2. Combine with encoder output via joint network
        3. Take argmax of output distribution
        4. If blank → advance to next time step
        5. If non-blank → append token, repeat at same time step
        6. Limit max symbols per step to prevent infinite loops

    Args:
        model: ConformerRNNT model (in eval mode)
        encoder_out: (T, D) — single utterance encoder output
        tokenizer: MarathiCharTokenizer for ID-to-text conversion
        max_symbols_per_step: Max non-blank emissions per time step

    Returns:
        Decoded text string
    """
    device = encoder_out.device
    T = encoder_out.size(0)

    # Initialize prediction network with blank token
    hidden = model.predictor.init_hidden(1, device)
    last_token = torch.tensor([[model.blank_id]], dtype=torch.long, device=device)

    output_tokens = []

    for t in range(T):
        enc_t = encoder_out[t].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        symbols_emitted = 0

        while symbols_emitted < max_symbols_per_step:
            # Run prediction network on last token
            pred_out, next_hidden = model.predictor(last_token, hidden)  # (1, 1, H)

            # Joint network
            logits = model.joint(enc_t, pred_out)  # (1, 1, 1, V)
            log_probs = F.log_softmax(logits.squeeze(), dim=-1)  # (V,)

            # Greedy: take argmax
            token_id = log_probs.argmax().item()

            if token_id == model.blank_id:
                # Blank: advance to next time step
                break
            else:
                # Non-blank: emit token and update hidden state
                output_tokens.append(token_id)
                last_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                hidden = next_hidden
                symbols_emitted += 1

    return tokenizer.decode(output_tokens)


@torch.no_grad()
def greedy_decode_batch(
    model,
    features: torch.Tensor,
    feature_lengths: torch.Tensor,
    tokenizer: MarathiCharTokenizer,
) -> list[str]:
    """
    Greedy decode a batch of utterances.
    
    Args:
        model: ConformerRNNT model
        features: (B, T, n_mels)
        feature_lengths: (B,)
        tokenizer: MarathiCharTokenizer
    
    Returns:
        List of decoded text strings
    """
    model.eval()
    encoder_out, enc_lengths = model.encode(features, feature_lengths)

    results = []
    for i in range(features.size(0)):
        length = enc_lengths[i].item()
        text = greedy_decode(model, encoder_out[i, :length], tokenizer)
        results.append(text)

    return results
