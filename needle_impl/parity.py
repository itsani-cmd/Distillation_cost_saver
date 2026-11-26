"""Numerical parity smoke test: pretrained T5-small vs its Needle port.

Run before fine-tuning:
    python -m needle_impl.parity --model t5-small
"""
import argparse
import numpy as np
import torch
import needle as ndl
from transformers import AutoTokenizer, T5ForConditionalGeneration
from .model import NeedleT5, StudentConfig
from .weights import import_hf_t5


def compare(model_name="t5-small", tolerance=1e-3):
    hf = T5ForConditionalGeneration.from_pretrained(model_name).eval().cpu()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = StudentConfig.from_hf(hf.config, dropout_rate=0.0, lora_dropout=0.0)
    student = NeedleT5(config, device=ndl.cpu())
    import_hf_t5(student, hf)
    student.eval()

    example = tokenizer(["translate English to German: The cat sleeps.",
                         "translate English to German: Good morning."],
                        padding=True, return_tensors="pt")
    start = hf.config.decoder_start_token_id
    decoder_ids = torch.tensor([[start, 15, 16], [start, 24, 25]], dtype=torch.long)
    with torch.no_grad():
        reference = hf(input_ids=example["input_ids"],
                       attention_mask=example["attention_mask"],
                       decoder_input_ids=decoder_ids).logits.cpu().numpy()
    results = student(
        ndl.Tensor(example["input_ids"].numpy().astype("float32"), device=ndl.cpu()),
        ndl.Tensor(decoder_ids.numpy().astype("float32"), device=ndl.cpu()),
        attention_mask=example["attention_mask"].numpy()
    )["logits"].numpy()
    absolute = np.abs(reference - results)
    print("reference shape:", reference.shape)
    print("max absolute error:", absolute.max())
    print("mean absolute error:", absolute.mean())
    np.testing.assert_allclose(results, reference, atol=tolerance, rtol=tolerance)
    print("PASS: Needle reproduces HF T5 logits within tolerance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="t5-small")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    compare(args.model, args.tolerance)
