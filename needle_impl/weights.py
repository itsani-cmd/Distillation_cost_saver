"""Load original Hugging Face T5 weights into the Needle student (strict mapping)."""
import numpy as np
import needle as ndl


def _copy(param, tensor, transpose=False):
    array = tensor.detach().cpu().numpy().astype("float32")
    if transpose:
        array = array.T
    if tuple(param.shape) != array.shape:
        raise ValueError(f"Weight shape mismatch: Needle {param.shape}, HF {array.shape}")
    # Assign to the existing Parameter's data to preserve module registration.
    param.data = ndl.Tensor(array, device=param.device, dtype=param.dtype, requires_grad=False)


def _linear(layer, state, prefix):
    _copy(layer.weight, state[prefix + ".weight"], transpose=True)


def _attention(attention, state, prefix):
    for part in ("q", "k", "v", "o"):
        _linear(getattr(attention, part), state, prefix + "." + part)
    if attention.relative_attention_bias is not None:
        _copy(attention.relative_attention_bias.weight,
              state[prefix + ".relative_attention_bias.weight"])


def _stack(stack, state, prefix, decoder):
    for i, block in enumerate(stack.layers):
        root = f"{prefix}.block.{i}.layer"
        _copy(block.self_norm.weight, state[root + ".0.layer_norm.weight"])
        _attention(block.self_attn, state, root + ".0.SelfAttention")
        ff_index = 2 if decoder else 1
        if decoder:
            _copy(block.cross_norm.weight, state[root + ".1.layer_norm.weight"])
            _attention(block.cross_attn, state, root + ".1.EncDecAttention")
        _copy(block.ff_norm.weight, state[f"{root}.{ff_index}.layer_norm.weight"])
        ff_root = f"{root}.{ff_index}.DenseReluDense"
        _linear(block.ff.wi, state, ff_root + ".wi")
        _linear(block.ff.wo, state, ff_root + ".wo")
    _copy(stack.final_layer_norm.weight, state[f"{prefix}.final_layer_norm.weight"])


def import_hf_t5(student, hf_model):
    """Import pretrained weights. LoRA A/B remain at zero-update initialization."""
    state = hf_model.state_dict()
    _copy(student.shared.weight, state["shared.weight"])
    _stack(student.encoder, state, "encoder", decoder=False)
    _stack(student.decoder, state, "decoder", decoder=True)
    # Keep all pretrained weights out of autograd parameter updates.
    # The LoRA factors are the only trainable student parameters in P0.
    for parameter in student.parameters():
        parameter.requires_grad = False
    for parameter in trainable_lora_parameters(student):
        parameter.requires_grad = True
    return student


def trainable_lora_parameters(student):
    """Optimizer parameters: ONLY LoRA factors in encoder and decoder attention."""
    result = []
    for stack in (student.encoder, student.decoder):
        for block in stack.layers:
            modules = [block.self_attn]
            if block.decoder:
                modules.append(block.cross_attn)
            for attention in modules:
                for name in ("q", "k", "v", "o"):
                    result.extend(getattr(attention, name).trainable_parameters())
    return result
