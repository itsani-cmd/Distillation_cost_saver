"""T5 encoder-decoder student in Needle, with shared embedding/output weights."""
from dataclasses import dataclass
import numpy as np
import needle as ndl
from needle import nn
from .transformer import T5Stack, attention_mask


@dataclass
class StudentConfig:
    vocab_size: int = 32128
    d_model: int = 512
    d_ff: int = 2048
    d_kv: int = 64
    num_heads: int = 8
    num_layers: int = 6
    num_decoder_layers: int = 6
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    pad_token_id: int = 0
    decoder_start_token_id: int = 0
    tie_word_embeddings: bool = True
    feed_forward_proj: str = "relu"

    @classmethod
    def from_hf(cls, hf, **kwargs):
        if hf.feed_forward_proj != "relu" or not hf.tie_word_embeddings:
            raise ValueError("P0 supports original T5 ReLU FFN and tied embeddings only")
        values = {k: getattr(hf, k) for k in cls.__dataclass_fields__ if hasattr(hf, k)}
        values.update(kwargs)
        return cls(**values)


class NeedleT5(nn.Module):
    """Teacher-forced T5 logits; the HF teacher is not part of this module."""

    def __init__(self, config, device=None, dtype="float32"):
        super().__init__()
        self.config = config
        self.device, self.dtype = device, dtype
        self.shared = nn.Embedding(config.vocab_size, config.d_model, device=device, dtype=dtype)
        self.encoder = T5Stack(config, decoder=False, device=device, dtype=dtype)
        self.decoder = T5Stack(config, decoder=True, device=device, dtype=dtype)

    def encode(self, input_ids, attention_mask=None):
        ids = input_ids.numpy()
        valid = (ids != self.config.pad_token_id) if attention_mask is None else np.asarray(attention_mask, bool)
        source_mask = attention_mask_fn(valid, ids.shape[1], False, self.device, self.dtype)
        return self.encoder(self.shared(input_ids), source_mask)

    def project(self, hidden):
        if self.config.tie_word_embeddings:
            hidden = hidden * (self.config.d_model ** -0.5)
        return hidden @ self.shared.weight.transpose((0, 1))

    def forward(self, input_ids, decoder_input_ids, attention_mask=None,
                decoder_attention_mask=None, return_all_exits=False):
        source_ids = np.asarray(input_ids.numpy())
        target_ids = np.asarray(decoder_input_ids.numpy())
        source_valid = (source_ids != self.config.pad_token_id) if attention_mask is None else np.asarray(attention_mask, bool)
        target_valid = np.ones_like(target_ids, dtype=bool) if decoder_attention_mask is None else np.asarray(decoder_attention_mask, bool)
        # HF T5 without decoder_attention_mask uses causal masking only.
        src_mask = attention_mask_fn(source_valid, source_ids.shape[1], False, self.device, self.dtype)
        tgt_mask = attention_mask_fn(target_valid, target_ids.shape[1], True, self.device, self.dtype)
        encoder_hidden = self.encoder(self.shared(input_ids), src_mask)
        decoded = self.decoder(self.shared(decoder_input_ids), tgt_mask, encoder_hidden,
                               src_mask, return_all_exits)
        if return_all_exits:
            _, exit_states = decoded
            logits = [self.project(h) for h in exit_states]
            return {"logits": logits[-1], "exit_logits": logits}
        return {"logits": self.project(decoded)}


def attention_mask_fn(valid, query_length, causal, device, dtype):
    return attention_mask(valid, query_length, causal, device, dtype)
