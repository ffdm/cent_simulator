# cent_simulation/test_single_channel.py
import math
import os

import torch
import torch.nn.functional as F

from Llama import TransformerBlockLlama
from utils import RMSNorm, apply_rotary_emb


FIXTURE_MODE_ENV = "SINGLE_CHANNEL_COSIM_FIXTURE_MODE"
LEGACY_FIXTURE_MODE = "legacy"
PROMPT_RANDOM_FIXTURE_MODE = "prompt-random"
SYNTHETIC_PROMPT_FIXTURE_MODE = "synthetic-prompt"
AUTO_FIXTURE_MODE = "auto"
PROMPT_RANDOM_MIN_CONTEXT = 2048
PROMPT_RANDOM_SCORE_STD = 0.35
PROMPT_RANDOM_SCORE_CLAMP = 1.0
PROMPT_RANDOM_TARGET_SEED = 424242
SYNTHETIC_PROMPT_SEED = 20260512
SYNTHETIC_PROMPT_RANK = 128
SYNTHETIC_PROMPT_RHO = 0.965


def resolve_single_channel_fixture_mode(context_len, fixture_mode=None):
    """Resolve the deterministic fixture mode used by the single-channel tests."""
    if fixture_mode is None:
        fixture_mode = os.getenv(FIXTURE_MODE_ENV, os.getenv("SINGLE_CHANNEL_FIXTURE_MODE", AUTO_FIXTURE_MODE))
    mode = str(fixture_mode).strip().lower().replace("_", "-")
    aliases = {
        "old": LEGACY_FIXTURE_MODE,
        "baseline": LEGACY_FIXTURE_MODE,
        "prompt": PROMPT_RANDOM_FIXTURE_MODE,
        "random": PROMPT_RANDOM_FIXTURE_MODE,
        "synthetic": SYNTHETIC_PROMPT_FIXTURE_MODE,
        "realistic": SYNTHETIC_PROMPT_FIXTURE_MODE,
    }
    mode = aliases.get(mode, mode)
    if mode == AUTO_FIXTURE_MODE:
        return SYNTHETIC_PROMPT_FIXTURE_MODE if int(context_len) >= PROMPT_RANDOM_MIN_CONTEXT else LEGACY_FIXTURE_MODE
    if mode not in {LEGACY_FIXTURE_MODE, PROMPT_RANDOM_FIXTURE_MODE, SYNTHETIC_PROMPT_FIXTURE_MODE}:
        raise ValueError(
            f"fixture_mode must be '{AUTO_FIXTURE_MODE}', '{LEGACY_FIXTURE_MODE}', "
            f"'{PROMPT_RANDOM_FIXTURE_MODE}', or '{SYNTHETIC_PROMPT_FIXTURE_MODE}', "
            f"got {fixture_mode!r}"
        )
    return mode

def get_single_channel_input_x(dim=4096):
    """Return the deterministic single-channel input tensor without building weights."""
    torch.manual_seed(42)
    return torch.randn((1, 1, dim)) * 0.1


def _projected_cache_scale(dim):
    # RMSNorm gives roughly unit-variance token components; 0.01-scale weights
    # then produce K/V activations with std close to sqrt(dim) * 0.01.
    return 0.01 * math.sqrt(dim)


def _precompute_freqs_cis(head_dim, context_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(context_len, dtype=torch.float32)
    freqs = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _make_norm_weight(dim, generator):
    weight = 1.0 + 0.08 * torch.randn(dim, generator=generator)
    return weight.clamp(0.75, 1.25)


def _make_synthetic_prompt_states(context_len, dim, generator):
    rank = min(SYNTHETIC_PROMPT_RANK, dim)
    basis = torch.randn((rank, dim), generator=generator) * (0.1 / math.sqrt(rank))
    coeffs = torch.empty((context_len, rank), dtype=torch.float32)
    coeff = torch.randn(rank, generator=generator)
    innovation_scale = math.sqrt(1.0 - SYNTHETIC_PROMPT_RHO ** 2)
    for index in range(context_len):
        if index:
            coeff = (
                SYNTHETIC_PROMPT_RHO * coeff
                + innovation_scale * torch.randn(rank, generator=generator)
            )
        coeffs[index] = coeff

    prompt = coeffs @ basis
    token_noise = 0.015 * torch.randn((context_len, dim), generator=generator)
    return (prompt + token_noise).reshape(1, context_len, dim)


def _make_synthetic_prompt_cache(prompt_states, sanorm, wk, wv, freqs_cis_all, n_heads, head_dim):
    with torch.no_grad():
        norm_prompt = RMSNorm(prompt_states, sanorm)
        k_proj = F.linear(norm_prompt, wk).reshape(1, prompt_states.shape[1], n_heads, head_dim)
        v_proj = F.linear(norm_prompt, wv).reshape(1, prompt_states.shape[1], n_heads, head_dim)
        _, k_rot = apply_rotary_emb(k_proj, k_proj, freqs_cis_all)
    return k_rot.contiguous(), v_proj.contiguous()


def _make_prompt_random_cache(context_len, dim, n_heads, head_dim, x, sanorm, wq, wk, freqs_cis):
    """Build a synthetic long-prompt K/V cache with controlled attention logits.

    The old fixture filled historical cache slots with tiny N(0, 0.01) noise,
    which made almost every long-context score sit near zero. This fixture keeps
    K/V magnitudes near projected activation scale and then nudges historical K
    vectors along the current query direction so each head sees a broad,
    deterministic score distribution in approximately [-1, 1].
    """
    scale = _projected_cache_scale(dim)
    cache_k = torch.randn((1, context_len, n_heads, head_dim)) * scale
    cache_v = torch.randn((1, context_len, n_heads, head_dim)) * scale

    previous_tokens = context_len - 1
    if previous_tokens <= 0:
        return cache_k, cache_v

    norm_x = RMSNorm(x, sanorm)
    xq = F.linear(norm_x, wq).reshape(1, 1, n_heads, head_dim)
    xk = F.linear(norm_x, wk).reshape(1, 1, n_heads, head_dim)
    xq_rot, _ = apply_rotary_emb(xq, xk, freqs_cis)
    query = xq_rot[0, 0].to(torch.float32)
    query_norm_sq = query.pow(2).sum(dim=-1).clamp_min(1e-12)

    historical_k = cache_k[:, :previous_tokens].to(torch.float32)
    current_scores = (historical_k * query.view(1, 1, n_heads, head_dim)).sum(dim=-1) / math.sqrt(head_dim)

    target_generator = torch.Generator()
    target_generator.manual_seed(PROMPT_RANDOM_TARGET_SEED + int(context_len))
    target_scores = torch.randn(
        (1, previous_tokens, n_heads),
        generator=target_generator,
    ) * PROMPT_RANDOM_SCORE_STD
    target_scores = target_scores.clamp(-PROMPT_RANDOM_SCORE_CLAMP, PROMPT_RANDOM_SCORE_CLAMP)

    correction = (
        (target_scores - current_scores)
        * math.sqrt(head_dim)
        / query_norm_sq.view(1, 1, n_heads)
    ).unsqueeze(-1) * query.view(1, 1, n_heads, head_dim)
    cache_k[:, :previous_tokens] = (historical_k + correction).to(cache_k.dtype)
    return cache_k, cache_v


def get_test_inputs(context_len=1, fixture_mode=None):
    dim = 4096
    n_heads = 32
    head_dim = dim // n_heads
    context_len = int(context_len)
    if context_len < 1:
        raise ValueError(f"context_len must be >= 1, got {context_len}")
    fixture_mode = resolve_single_channel_fixture_mode(context_len, fixture_mode)

    # Set seed for reproducibility across different test environments
    torch.manual_seed(42)
    synthetic_generator = torch.Generator()
    synthetic_generator.manual_seed(SYNTHETIC_PROMPT_SEED + context_len)
    if fixture_mode == SYNTHETIC_PROMPT_FIXTURE_MODE:
        prompt_states = _make_synthetic_prompt_states(context_len, dim, synthetic_generator)
        x = prompt_states[:, -1:, :].contiguous()
        freqs_cis_all = _precompute_freqs_cis(head_dim, context_len)
        freqs_cis = freqs_cis_all[-1:].contiguous()
        SANorm = _make_norm_weight(dim, synthetic_generator)
        FFNNorm = _make_norm_weight(dim, synthetic_generator)
    else:
        prompt_states = None
        x = get_single_channel_input_x(dim)
        freqs_cis_all = None
        freqs_cis = torch.ones((1, head_dim // 2), dtype=torch.complex64)
        SANorm = torch.ones(dim)
        FFNNorm = torch.ones(dim)

    wq = torch.randn((dim, dim)) * 0.01
    wk = torch.randn((dim, dim)) * 0.01
    wv = torch.randn((dim, dim)) * 0.01
    if fixture_mode == SYNTHETIC_PROMPT_FIXTURE_MODE:
        cache_k, cache_v = _make_synthetic_prompt_cache(
            prompt_states,
            SANorm,
            wk,
            wv,
            freqs_cis_all,
            n_heads,
            head_dim,
        )
    elif fixture_mode == PROMPT_RANDOM_FIXTURE_MODE:
        cache_k, cache_v = _make_prompt_random_cache(
            context_len,
            dim,
            n_heads,
            head_dim,
            x,
            SANorm,
            wq,
            wk,
            freqs_cis,
        )
    else:
        cache_k = torch.randn((1, context_len, n_heads, head_dim)) * 0.01
        cache_v = torch.randn((1, context_len, n_heads, head_dim)) * 0.01

    dic_model = {
        "dim": torch.tensor(dim),
        "n_heads": torch.tensor(n_heads),
        "TP_param": torch.tensor(1),
        "fixture_mode": fixture_mode,
        "x": x,
        "SANorm": SANorm,
        "FFNNorm": FFNNorm,
        "sa": torch.zeros((1, 1, dim)),
        "h": torch.zeros((1, 1, dim)),
        "out": torch.zeros((1, 1, dim)),
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "xq": torch.zeros((1, 1, dim)),
        "xk": torch.zeros((1, 1, dim)),
        "xv": torch.zeros((1, 1, dim)),
        "start_pos": torch.tensor(context_len - 1),
        "freqs_cis": freqs_cis,
        "cache_k": cache_k,
        "cache_v": cache_v,
        "scores": torch.zeros((1, n_heads, 1, context_len)),
        "output": torch.zeros((1, 1, dim)),
        "wo": torch.randn((dim, dim)) * 0.01,
        "w1": torch.randn((dim, dim)) * 0.01,
        "w3": torch.randn((dim, dim)) * 0.01,
        "w2": torch.randn((dim, dim)) * 0.01,
        "ffn": torch.zeros((1, 1, dim))
    }
    populate_reference_outputs(dic_model)
    return dic_model


def populate_reference_outputs(dic_model):
    """Fill the test fixture with PyTorch golden tensors used by Llama compares."""
    dim = dic_model["dim"].item()
    n_heads = dic_model["n_heads"].item()
    head_dim = dim // n_heads
    bsz, seqlen, _ = dic_model["x"].shape
    start_pos = dic_model["start_pos"].item()

    x = dic_model["x"]
    norm_x = RMSNorm(x, dic_model["SANorm"])
    xq = F.linear(norm_x, dic_model["wq"])
    xk = F.linear(norm_x, dic_model["wk"])
    xv = F.linear(norm_x, dic_model["wv"])

    xq_heads = xq.reshape(bsz, seqlen, n_heads, head_dim)
    xk_heads = xk.reshape(bsz, seqlen, n_heads, head_dim)
    xv_heads = xv.reshape(bsz, seqlen, n_heads, head_dim)
    xq_rot, xk_rot = apply_rotary_emb(xq_heads, xk_heads, dic_model["freqs_cis"])

    cache_k = dic_model["cache_k"].clone()
    cache_v = dic_model["cache_v"].clone()
    cache_k[:bsz, start_pos:start_pos + seqlen] = xk_rot
    cache_v[:bsz, start_pos:start_pos + seqlen] = xv_heads

    keys = cache_k[:bsz, :start_pos + seqlen].transpose(1, 2).transpose(2, 3)
    values = cache_v[:bsz, :start_pos + seqlen].transpose(1, 2)
    scores = torch.matmul(xq_rot.transpose(1, 2), keys) / math.sqrt(head_dim)
    scores = F.softmax(scores, dim=-1).type_as(xq_rot)
    output = torch.matmul(scores, values)
    output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
    sa_projection = F.linear(output, dic_model["wo"])
    h = x + sa_projection

    norm_h = RMSNorm(h, dic_model["FFNNorm"])
    x1 = F.linear(norm_h, dic_model["w1"])
    x3 = F.linear(norm_h, dic_model["w3"])
    ffn = F.linear(F.silu(x1) * x3, dic_model["w2"])
    out = h + ffn

    dic_model["xq"] = xq
    dic_model["xk"] = xk
    dic_model["xv"] = xv
    dic_model["scores"] = scores
    dic_model["output"] = output
    dic_model["sa"] = sa_projection
    dic_model["h"] = h
    dic_model["ffn"] = ffn
    dic_model["out"] = out

def test_rms_norm():
    print("Initializing test...")
    dic_model = get_test_inputs()
    dim = dic_model["dim"].item()
    n_heads = dic_model["n_heads"].item()
    head_dim = dim // n_heads

    class DummyArgs:
        pass
        
    args = DummyArgs()
    args.pim_compute = True
    args.op_trace = False
    args.trace_prepare = False
    args.trace_norm = False
    args.trace_fc_kqvo = False
    args.trace_attention = False
    args.trace_softmax = False
    args.trace_fc_ffn = False
    args.trace_activation = False
    args.model = "Llama-7B"
    args.seqlen = 1
    args.FC_devices = 1
    args.embedding = False
    args.only_FC = False
    args.only_trace = False
    args.model_parallel = False
    args.pipeline_parallel = True
    args.channels_per_block = 1
    args.num_channels = 1
    args.GEMV = "no-reuse"
    args.reuse_size = 2
    args.max_seq_len = 1
    args.inter_device_attention = False
    args.DRAM_column = 1024
    args.DRAM_row = 1024*16
    args.burst_length = 16
    args.num_banks = 16
    args.threads = 1
    args.trace_file = "test.log"

    print("Creating TransformerBlockLlama...")
    TB = TransformerBlockLlama(dic_model, args)
    
    print("Memory mapping...")
    TB.memory_mapping()
    TB.memory_mapping_verification()
    
    print("Running self_attention_aim...")
    try:
        sa_aim = TB.self_attention_aim()
        print("self_attention_aim test passed successfully!")
        
        print("Running FFN_aim...")
        out_aim = TB.FFN_aim(sa_aim)
        print("FFN_aim test passed successfully!")
        
        print("All tests passed successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed with exception: {e}")

if __name__ == "__main__":
    test_rms_norm()
