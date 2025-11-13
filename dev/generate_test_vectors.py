"""
Generate test vectors for validating llm.clj implementations against PyTorch.

This script creates ground truth test data by running operations in PyTorch
and saving inputs/outputs in EDN format (Clojure's data notation).

Usage:
    python dev/generate_test_vectors.py

Output:
    dev/test_vectors/*.edn - Test data files for each operation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


def to_edn_value(value):
    """Convert Python value to EDN literal string."""
    if isinstance(value, (int, float)):
        return str(float(value))
    elif hasattr(value, 'item'):  # NumPy scalar types
        return str(float(value.item()))
    elif isinstance(value, list):
        elements = [to_edn_value(v) for v in value]
        return '[' + ' '.join(elements) + ']'
    elif isinstance(value, str):
        return f'"{value}"'
    else:
        raise ValueError(f"Unsupported type for EDN: {type(value)}")


def tensor_to_edn(tensor):
    """Convert PyTorch tensor to EDN nested vector."""
    # Detach from computation graph before converting to numpy
    tensor = tensor.detach()
    
    if tensor.dim() == 1:
        return to_edn_value(list(tensor.float().numpy()))
    elif tensor.dim() == 2:
        rows = [list(row.float().numpy()) for row in tensor]
        return to_edn_value(rows)
    elif tensor.dim() == 3:
        depth = [[list(col.float().numpy()) for col in row] for row in tensor]
        return to_edn_value(depth)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")


def write_edn(data_dict, filepath):
    """Write dictionary to EDN file with proper Clojure syntax."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(';; Auto-generated test vectors from PyTorch\n')
        f.write(';; Do not edit manually - regenerate using dev/generate_test_vectors.py\n\n')
        f.write('{')
        
        for i, (key, value) in enumerate(data_dict.items()):
            if i > 0:
                f.write('\n ')
            f.write(f':{key} {value}')
        
        f.write('}\n')


def generate_matmul_test():
    """Generate test vectors for matrix multiplication."""
    torch.manual_seed(42)
    
    # Test case: small (easy to verify manually)
    B, T, C, OC = 2, 2, 3, 4
    inp = torch.randn(B, T, C)
    weight = torch.randn(OC, C)
    bias = torch.randn(OC)
    
    # Forward pass
    inp_2d = inp.reshape(B * T, C)
    out = F.linear(inp_2d, weight, bias)
    
    # Backward pass
    inp_2d.requires_grad_(True)
    weight_param = nn.Parameter(weight.clone())
    bias_param = nn.Parameter(bias.clone())
    
    out_auto = F.linear(inp_2d, weight_param, bias_param)
    dout = torch.ones_like(out)  # Simple gradient for easier verification
    out_auto.backward(dout)
    
    edn_data = {
        'operation': '"matmul"',
        'test-case': '"small"',
        'inputs': '{:inp ' + tensor_to_edn(inp_2d) + 
                  ' :weight ' + tensor_to_edn(weight) +
                  ' :bias ' + tensor_to_edn(bias) + '}',
        'expected': '{:forward ' + tensor_to_edn(out) +
                    ' :dinp ' + tensor_to_edn(inp_2d.grad) +
                    ' :dweight ' + tensor_to_edn(weight_param.grad) +
                    ' :dbias ' + tensor_to_edn(bias_param.grad) + '}'
    }
    
    write_edn(edn_data, Path("dev/test_vectors/matmul_small.edn"))
    print("✓ Generated matmul test vectors")


def generate_gelu_test():
    """Generate test vectors for GELU activation."""
    torch.manual_seed(44)
    
    x = torch.randn(4, 8)
    
    # Forward
    out = F.gelu(x)
    
    # Backward
    x.requires_grad_(True)
    out_auto = F.gelu(x)
    dout = torch.ones_like(out)
    out_auto.backward(dout)
    
    edn_data = {
        'operation': '"gelu"',
        'test-case': '"standard"',
        'inputs': '{:x ' + tensor_to_edn(x) + '}',
        'expected': '{:forward ' + tensor_to_edn(out) +
                    ' :dx ' + tensor_to_edn(x.grad) + '}'
    }
    
    write_edn(edn_data, Path("dev/test_vectors/gelu_standard.edn"))
    print("✓ Generated GELU test vectors")


def generate_encoder_test():
    """Generate test vectors for encoder (embedding lookup)."""
    torch.manual_seed(46)
    
    # Small vocabulary and embedding dimension for testing
    vocab_size = 10
    max_seq_len = 8
    C = 16  # embedding dimension
    B = 2   # batch size
    T = 4   # sequence length
    
    # Create embedding tables
    wte = torch.randn(vocab_size, C)  # token embeddings
    wpe = torch.randn(max_seq_len, C)  # position embeddings
    
    # Random token indices
    inp = torch.randint(0, vocab_size, (B, T))
    
    # Forward: lookup and add embeddings
    # For each batch and position, lookup token embedding and add position embedding
    out = torch.zeros(B, T, C)
    for b in range(B):
        for t in range(T):
            token_idx = inp[b, t].item()
            out[b, t] = wte[token_idx] + wpe[t]
    
    # Backward: accumulate gradients to embeddings
    dout = torch.ones(B, T, C)  # Gradient from upstream
    
    # Initialize gradients
    dwte = torch.zeros_like(wte)
    dwpe = torch.zeros_like(wpe)
    
    # Accumulate gradients
    for b in range(B):
        for t in range(T):
            token_idx = inp[b, t].item()
            dwte[token_idx] += dout[b, t]
            dwpe[t] += dout[b, t]
    
    edn_data = {
        'operation': '"encoder"',
        'test-case': '"small"',
        'inputs': '{:inp ' + tensor_to_edn(inp) + 
                  ' :wte ' + tensor_to_edn(wte) +
                  ' :wpe ' + tensor_to_edn(wpe) + '}',
        'expected': '{:forward ' + tensor_to_edn(out) +
                    ' :dwte ' + tensor_to_edn(dwte) +
                    ' :dwpe ' + tensor_to_edn(dwpe) + '}'
    }
    
    write_edn(edn_data, Path("dev/test_vectors/encoder_small.edn"))
    print("✓ Generated encoder test vectors")


def main():
    """Generate all test vectors."""
    print("Generating test vectors from PyTorch...")
    print()
    
    generate_matmul_test()
    generate_gelu_test()
    generate_encoder_test()
    
    print()
    print("✓ Test vectors generated successfully!")
    print(f"  Output directory: dev/test_vectors/")
    print()
    print("Next: Run validation tests in Clojure:")
    print("  lein test llm.neo.validation-test")


if __name__ == "__main__":
    main()