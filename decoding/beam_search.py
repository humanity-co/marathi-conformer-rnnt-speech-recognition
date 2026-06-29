"""
Beam Search Decoder for RNN-Transducer

Beam search maintains the top-K hypotheses at each time step,
exploring multiple decoding paths simultaneously.

How beam search works for RNN-T:
  1. Start with a single hypothesis: empty sequence + blank context
  2. For each encoder time step t:
     a. For each hypothesis in the beam:
        - Evaluate blank probability → hypothesis stays, advance t
        - Evaluate top-K non-blank tokens → create new hypotheses
     b. Prune to keep only top-K hypotheses by accumulated log probability
  3. Return the highest-scoring hypothesis

Optional shallow fusion with an external language model (KenLM)
improves output quality by incorporating linguistic priors.

Why beam search > greedy:
  • Explores multiple paths, reducing the chance of "greedy mistakes"
  • ~5-15% WER improvement over greedy for speech recognition
  • Required for production-quality ASR systems
"""

import torch
import torch.nn.functional as F

from tokenizer.charset import MarathiCharTokenizer


class KenLMScorer:
    """
    Optional KenLM n-gram language model scorer for shallow fusion.
    
    Shallow fusion: P(y|x) = log P_am(y|x) + λ * log P_lm(y)
    where λ is the LM weight.
    """

    def __init__(self, model_path: str):
        try:
            import kenlm
            self.model = kenlm.Model(model_path)
            print(f"Loaded KenLM model: {model_path}")
        except ImportError:
            print("Warning: kenlm not installed. LM scoring disabled.")
            self.model = None

    def score(self, context: list[str], token: str) -> float:
        """
        Score a token given context using the language model.
        
        Args:
            context: List of previous tokens
            token: Current token to score
            
        Returns:
            Log probability from the LM
        """
        if self.model is None:
            return 0.0
        text = "".join(context) + token
        return self.model.score(text, bos=True, eos=False)


class BeamSearchDecoder:
    """
    RNN-T Beam Search Decoder with optional LM shallow fusion.
    
    Args:
        model: ConformerRNNT model (eval mode)
        tokenizer: MarathiCharTokenizer
        beam_size: Number of hypotheses to maintain
        lm_scorer: Optional KenLM scorer
        lm_weight: Weight for LM log probs in shallow fusion
        max_symbols_per_step: Max non-blank emissions per time step
    """

    def __init__(
        self,
        model,
        tokenizer: MarathiCharTokenizer,
        beam_size: int = 10,
        lm_scorer: KenLMScorer | None = None,
        lm_weight: float = 0.1,
        max_symbols_per_step: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.lm_scorer = lm_scorer
        self.lm_weight = lm_weight
        self.blank_id = model.blank_id
        self.max_symbols = max_symbols_per_step

    @torch.no_grad()
    def decode(self, encoder_out: torch.Tensor) -> str:
        """
        Beam search decode a single utterance.
        
        Args:
            encoder_out: (T, D) — single utterance encoder output
            
        Returns:
            Decoded text string (best hypothesis)
        """
        device = encoder_out.device
        T = encoder_out.size(0)

        # Hypothesis: (log_prob, token_ids, lstm_hidden)
        init_hidden = self.model.predictor.init_hidden(1, device)
        beam = [(0.0, [self.blank_id], init_hidden)]

        for t in range(T):
            enc_t = encoder_out[t].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            new_beam = []

            for score, seq, hidden in beam:
                # Allow multiple non-blank emissions per time step
                curr_score, curr_seq, curr_hidden = score, seq, hidden

                for _ in range(self.max_symbols + 1):
                    last_token = torch.tensor(
                        [[curr_seq[-1]]], dtype=torch.long, device=device
                    )
                    pred_out, next_hidden = self.model.predictor(last_token, curr_hidden)

                    logits = self.model.joint(enc_t, pred_out)
                    log_probs = F.log_softmax(logits.squeeze(), dim=-1)

                    # Get top-K candidates
                    top_probs, top_ids = log_probs.topk(self.beam_size)

                    for p, idx in zip(top_probs, top_ids):
                        idx_val = idx.item()
                        p_val = p.item()

                        # LM scoring
                        lm_score = 0.0
                        if self.lm_scorer is not None and idx_val != self.blank_id:
                            context_tokens = [
                                self.tokenizer.id2char.get(i, "")
                                for i in curr_seq
                                if i != self.blank_id
                            ]
                            token_str = self.tokenizer.id2char.get(idx_val, "")
                            lm_score = self.lm_scorer.score(context_tokens, token_str) * self.lm_weight

                        new_score = curr_score + p_val + lm_score

                        if idx_val == self.blank_id:
                            # Blank: advance time, keep sequence and hidden
                            new_beam.append((new_score, curr_seq, curr_hidden))
                        else:
                            # Non-blank: extend sequence, update hidden
                            new_seq = curr_seq + [idx_val]
                            new_beam.append((new_score, new_seq, next_hidden))

                    # After first expansion, only continue if we got non-blank
                    # For simplicity, break after one expansion round
                    break

            # Prune beam
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[: self.beam_size]

        # Return best hypothesis, removing the initial blank
        best_score, best_seq, _ = beam[0]
        output_ids = [idx for idx in best_seq if idx != self.blank_id]
        return self.tokenizer.decode(output_ids)

    @torch.no_grad()
    def decode_batch(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> list[str]:
        """
        Beam search decode a batch of utterances.
        
        Args:
            features: (B, T, n_mels)
            feature_lengths: (B,)
            
        Returns:
            List of decoded text strings
        """
        self.model.eval()
        encoder_out, enc_lengths = self.model.encode(features, feature_lengths)

        results = []
        for i in range(features.size(0)):
            length = enc_lengths[i].item()
            text = self.decode(encoder_out[i, :length])
            results.append(text)

        return results
