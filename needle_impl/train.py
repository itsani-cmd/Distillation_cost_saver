"""P0: train Needle LoRA student using logits from a frozen PyTorch T5 teacher.

Run from the repository root:
    python -m needle_impl.train --config configs/needle_p0.yaml --max-steps 10
"""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import needle as ndl
from needle import optim
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration

from src.config import load_config
from src.data import build_dataloaders, build_tokenizer
from .model import NeedleT5, StudentConfig
from .weights import import_hf_t5, trainable_lora_parameters
from .losses import distillation_loss


def shifted_targets(labels, pad_id, decoder_start_id):
    """Labels use -100 on padding; decoder receives start + preceding labels."""
    labels = np.asarray(labels, dtype=np.int64)
    valid = labels != -100
    safe = np.where(valid, labels, pad_id).astype("float32")
    decoder = np.full_like(safe, pad_id)
    decoder[:, 0] = decoder_start_id
    decoder[:, 1:] = safe[:, :-1]
    return safe, decoder, valid.astype("float32")


def save_adapters(model, path):
    """Save only Needle LoRA factors; base weights are reloaded from HF on restart."""
    arrays = {}
    for stack_name in ("encoder", "decoder"):
        for i, block in enumerate(getattr(model, stack_name).layers):
            for attention_name in ("self_attn", "cross_attn"):
                if not hasattr(block, attention_name):
                    continue
                attention = getattr(block, attention_name)
                for projection in ("q", "k", "v", "o"):
                    linear = getattr(attention, projection)
                    root = f"{stack_name}.{i}.{attention_name}.{projection}"
                    arrays[root + ".A"] = linear.A.numpy()
                    arrays[root + ".B"] = linear.B.numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_adapters(model, path):
    """Restore LoRA adapters into an already initialized pretrained Needle model."""
    with np.load(path) as arrays:
        expected = set()
        for stack_name in ("encoder", "decoder"):
            for i, block in enumerate(getattr(model, stack_name).layers):
                for attention_name in ("self_attn", "cross_attn"):
                    if not hasattr(block, attention_name):
                        continue
                    attention = getattr(block, attention_name)
                    for projection in ("q", "k", "v", "o"):
                        linear = getattr(attention, projection)
                        root = f"{stack_name}.{i}.{attention_name}.{projection}"
                        for suffix in ("A", "B"):
                            key = root + "." + suffix
                            expected.add(key)
                            p = getattr(linear, suffix)
                            if tuple(p.shape) != arrays[key].shape:
                                raise ValueError(f"Bad shape in adapter file: {key}")
                            p.data = ndl.Tensor(arrays[key], device=p.device,
                                                dtype=p.dtype, requires_grad=False)
        if expected != set(arrays.files):
            raise ValueError("Adapter file keys don't match model structure")


def run(config_path, max_steps=None, output="outputs/needle_p0_lora.npz"):
    cfg = load_config(config_path)
    if not cfg.model.teacher_name_or_path or cfg.distillation.kd_weight <= 0:
        raise ValueError("P0 requires a frozen teacher and positive kd_weight")
    if cfg.distillation.kd_mode != "word":
        raise ValueError("P0 requires compatible-tokenizer word-level KD")
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    tokenizer = build_tokenizer(cfg.model.student_name_or_path)
    teacher_tokenizer = AutoTokenizer.from_pretrained(cfg.model.teacher_name_or_path)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Teacher/student tokenizers differ; token KL is invalid. Use matching T5-family models.")
    train_loader, _ = build_dataloaders(cfg, tokenizer)

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    needle_device = ndl.cuda() if torch_device == "cuda" else ndl.cpu()
    reference = T5ForConditionalGeneration.from_pretrained(cfg.model.student_name_or_path)
    student_cfg = StudentConfig.from_hf(reference.config, lora_rank=cfg.lora.r,
                                        lora_alpha=cfg.lora.alpha,
                                        lora_dropout=cfg.lora.dropout)
    student = NeedleT5(student_cfg, device=needle_device)
    import_hf_t5(student, reference)
    del reference
    student.train()

    teacher = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.teacher_name_or_path)
    teacher.to(torch_device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    parameters = trainable_lora_parameters(student)
    if not parameters:
        raise RuntimeError("No trainable LoRA parameters")
    optimizer = optim.Adam(parameters, lr=cfg.training.learning_rate,
                           weight_decay=cfg.training.weight_decay)
    print(f"Needle LoRA tensors: {len(parameters)}. Teacher: {cfg.model.teacher_name_or_path}")
    step = 0
    for epoch in range(cfg.training.num_epochs):
        for batch in train_loader:
            safe, decoder_np, valid = shifted_targets(
                batch["labels"].numpy(), tokenizer.pad_token_id,
                student_cfg.decoder_start_token_id
            )
            source = batch["input_ids"]
            source_np = source.numpy().astype("float32")
            decoder = ndl.Tensor(decoder_np, device=needle_device, requires_grad=False)
            inputs = ndl.Tensor(source_np, device=needle_device, requires_grad=False)
            with torch.no_grad():
                target = teacher(
                    input_ids=source.to(torch_device),
                    attention_mask=batch["attention_mask"].to(torch_device),
                    decoder_input_ids=torch.as_tensor(decoder_np, dtype=torch.long,
                                                      device=torch_device),
                ).logits.detach().float().cpu().numpy()
            if target.shape[-1] != student_cfg.vocab_size:
                raise ValueError("Teacher and student vocab sizes differ")
            outputs = student(inputs, decoder, attention_mask=batch["attention_mask"].numpy(),
                              return_all_exits=True)
            loss = distillation_loss(outputs["exit_logits"], safe.astype("int64"), valid,
                                     target, cfg.distillation.kd_weight,
                                     cfg.distillation.temperature, needle_device)
            value = float(loss.numpy())
            if not np.isfinite(value):
                raise RuntimeError("Non-finite Needle loss")
            optimizer.reset_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step == 1 or step % cfg.training.log_every == 0:
                print(f"epoch={epoch} step={step} loss={value:.4f}")
            if max_steps and step >= max_steps:
                save_adapters(student, output)
                print(f"Saved {output}")
                return
        save_adapters(student, output)
    print(f"Saved {output}; total steps={step}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/needle_p0.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", default="outputs/needle_p0_lora.npz")
    args = parser.parse_args()
    run(args.config, args.max_steps, args.output)


if __name__ == "__main__":
    main()
