#!/usr/bin/env python3
"""
Script to release RAM and GPU memory.
Run this in a Python environment where you have PyTorch loaded.
"""

import gc
import sys

def clear_gpu_memory():
    """Clear PyTorch GPU cache if available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU Memory before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print(f"GPU Memory after: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print("✓ GPU cache cleared")
        else:
            print("No CUDA devices available")
    except ImportError:
        print("PyTorch not available")

def clear_python_memory():
    """Force Python garbage collection."""
    print("Running garbage collection...")
    collected = gc.collect()
    print(f"✓ Garbage collected {collected} objects")

if __name__ == "__main__":
    print("Clearing memory...")
    clear_python_memory()
    clear_gpu_memory()
    print("\nDone!")


