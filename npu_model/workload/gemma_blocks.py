import math

import torch
from torch import nn


def gelu_impl(x: torch.Tensor) -> torch.Tensor:
    return (
        x
        * 0.5
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


def norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Compute variance in float32 (like the source implementation)
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    # Compute normalization in float32
    normed_inputs = x * torch.rsqrt(var + eps)
    return normed_inputs


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py#L86
def compute_default_rope_parameters(head_dim: int = 256) -> tuple[torch.Tensor, float]:
    rope_theta = 10000.0

    base = rope_theta
    partial_rotary_factor = 1.0
    head_dim = head_dim
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
    )
    return inv_freq, attention_factor


def gemma_rms_norm_forward(
    x: torch.Tensor,
    eps: float = 1e-6,
    cond_dim: int | None = None,
):
    if cond_dim:
        raise NotImplementedError(
            "Warning: cond_dim is not supported yet, which is required by PI0.5"
        )

    # original dtype, could be half-precision
    dtype = x.dtype
    normed_inputs = norm(x, eps)

    return normed_inputs.to(dtype)


def gemma_mlp_gate_up_forward(
    x: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    use_gelu: bool = True,
) -> torch.Tensor:
    """
    Gate and up projection with gated multiply (GeGLU pre-down part).
    Returns x_gated = activation(x_gate) * x_up.
    use_gelu=False yields x_gate * x_up (simplified, no activation).
    Weight layout: (in_features, out_features) to match NPU matmul x @ w.
    """
    x_gate = torch.matmul(x.float(), gate_proj_weight.float())
    x_up = torch.matmul(x.float(), up_proj_weight.float())
    x_act = gelu_impl(x_gate) if use_gelu else x_gate
    x_gated = x_act * x_up
    return x_gated


def gemma_mlp_forward(
    x: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    act_fn = nn.GELU()

    x_gate = torch.matmul(x, gate_proj_weight.T)
    x_up = torch.matmul(x, up_proj_weight.T)
    x_act = gelu_impl(x_gate)
    x_gated = x_act * x_up
    x_down = torch.matmul(x_gated, down_proj_weight.T)
    return x_down


def gemma_rotary_embedding_forward(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    # max_position_embeddings: int,
    head_dim: int = 256,
):
    # max_seq_len_cached = config.max_position_embeddings
    # original_max_seq_len = config.max_position_embeddings

    inv_freq, attention_scaling = compute_default_rope_parameters(head_dim)

    inv_freq_expanded = (
        inv_freq[None, :, None]
        .float()
        .expand(position_ids.shape[0], -1, 1)
        .to(x.device)
    )
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            1, 2
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    num_key_value_groups: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float = 1.0,
):
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def gemma_attention_forward(
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    past_key_value: None = None,
    cache_position: None = None,
    head_dim: int = 256,
    num_attention_heads: int = 8,
    num_key_value_heads: int = 1,
    use_cache: bool = False,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = (
        torch.matmul(hidden_states, q_proj_weight.T).view(hidden_shape).transpose(1, 2)
    )
    key_states = (
        torch.matmul(hidden_states, k_proj_weight.T).view(hidden_shape).transpose(1, 2)
    )
    value_states = (
        torch.matmul(hidden_states, v_proj_weight.T).view(hidden_shape).transpose(1, 2)
    )

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Use cache if provided
    if past_key_value is not None:
        raise NotImplementedError("Warning: TODO: past_key_value not implemented yet")
    #     if use_cache:
    #         # sin and cos are specific to RoPE models; cache_position needed for the static cache
    #         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #         key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    #     else:
    #         key_states = torch.cat([past_key_value[self.layer_idx][0], key_states], dim=2)
    #         value_states = torch.cat([past_key_value[self.layer_idx][1], value_states], dim=2)

    scaling = head_dim**-0.5
    num_key_value_groups = num_attention_heads // num_key_value_heads

    attn_output, attn_weights = eager_attention_forward(
        num_key_value_groups,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=scaling,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = torch.matmul(attn_output, o_proj_weight.T)
    return attn_output, attn_weights

def gemma_layer_generic_fp8(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
):
    seq_len = hidden_states.shape[0]
    d_model = hidden_states.shape[1]
    d_head = q_proj_weight.shape[1]
    d_ff = gate_proj_weight.shape[1]

    # Convert to float32 first
    hidden_states = hidden_states.float()
    q_proj_weight = q_proj_weight.float()
    k_proj_weight = k_proj_weight.float()
    v_proj_weight = v_proj_weight.float()
    o_proj_weight = o_proj_weight.float()
    gate_proj_weight = gate_proj_weight.float()
    up_proj_weight = up_proj_weight.float()
    down_proj_weight = down_proj_weight.float()

    # 1. Apply rmsnorm to the input
    hs_norm = norm(hidden_states)

    # 2. Compute QKV
    Q = torch.matmul(hs_norm, q_proj_weight)
    K = torch.matmul(hs_norm, k_proj_weight)
    V = torch.matmul(hs_norm, v_proj_weight)

    # 3. Compute Attention. Note that we're assuming a single head for simplicity
    attn_scores = torch.matmul(Q, K.T) / (d_head ** 0.5)
    attn_probs = torch.nn.functional.softmax(attn_scores, dim=1)
    attn_out = torch.matmul(attn_probs, V)

    # # 4. Project back to d_model
    # attn_out = torch.matmul(attn_out, o_proj_weight)

    # # 5. Residual connection
    # hs_intermediate = hidden_states + attn_out

    # # 6. Apply rmsnorm to hs_intermediate
    # hs_norm = norm(hs_intermediate)

    # # 7. Perform gate/up projection
    # gate = torch.matmul(hs_norm, gate_proj_weight)
    # up = torch.matmul(hs_norm, up_proj_weight)

    # # 8. Apply GELU to gate
    # gate = gelu_impl(gate)

    # # 9. Element-wise multiply of gate and up
    # mlp_out = gate * up

    # # 10. Perform down projection
    # mlp_out = torch.matmul(mlp_out, down_proj_weight)

    # # 11. Residual connection
    # hs_final = hs_intermediate + mlp_out

    # Done
    return attn_out.to(torch.float8_e4m3fn)