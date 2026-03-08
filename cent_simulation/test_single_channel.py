# cent_simulation/test_single_channel.py
import torch
import sys
import argparse
from Llama import TransformerBlockLlama
from utils import compare

def test_rms_norm():
    print("Initializing test...")
    dim = 4096
    n_heads = 32
    head_dim = dim // n_heads

    dic_model = {
        "dim": torch.tensor(dim),
        "n_heads": torch.tensor(n_heads),
        "TP_param": torch.tensor(1),
        "x": torch.randn((1, 1, dim)) * 0.1,
        "SANorm": torch.ones(dim),
        "FFNNorm": torch.ones(dim),
        "sa": torch.zeros((1, 1, dim)),
        "h": torch.zeros((1, 1, dim)),
        "out": torch.zeros((1, 1, dim)),
        "wq": torch.randn((dim, dim)) * 0.01,
        "wk": torch.randn((dim, dim)) * 0.01,
        "wv": torch.randn((dim, dim)) * 0.01,
        "xq": torch.zeros((1, 1, dim)),
        "xk": torch.zeros((1, 1, dim)),
        "xv": torch.zeros((1, 1, dim)),
        "start_pos": torch.tensor(0),
        "freqs_cis": torch.ones((1, head_dim // 2), dtype=torch.complex64),
        "cache_k": torch.zeros((1, 1, n_heads, head_dim)),
        "cache_v": torch.zeros((1, 1, n_heads, head_dim)),
        "scores": torch.zeros((1, n_heads, 1, 1)),
        "output": torch.zeros((1, 1, dim)),
        "wo": torch.randn((dim, dim)) * 0.01,
        "w1": torch.randn((dim, dim)) * 0.01,
        "w3": torch.randn((dim, dim)) * 0.01,
        "w2": torch.randn((dim, dim)) * 0.01,
        "ffn": torch.zeros((1, 1, dim))
    }

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
