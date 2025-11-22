# benchmarks/import_time/benchmark_configs_import.py
"""
Benchmark script to measure the improvement from lazy loading configs.

This script compares:
1. Eager loading (old __init__bk.py style) vs Lazy loading (new __init__.py)
2. Cold import time (fresh process)
3. Warm import time (already imported)
4. Time to access a specific config (lazy loading overhead)
5. Impact on actual vLLM startup
"""
from __future__ import annotations
import subprocess
import sys
import time
import statistics
from typing import List
from pathlib import Path

# Number of samples for statistical significance
N_SAMPLES = 20


def measure_cold_import(import_statement: str) -> List[float]:
    """Measure import time in a fresh Python process."""
    samples: List[float] = []
    
    cmd = [
        sys.executable,
        "-c",
        (
            "import time; "
            "t0 = time.perf_counter(); "
            f"{import_statement}; "
            "t1 = time.perf_counter(); "
            "print(t1 - t0)"
        )
    ]
    
    print(f"Measuring cold import: {import_statement}")
    for i in range(N_SAMPLES):
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = float(result.stdout.strip())
        samples.append(elapsed)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{N_SAMPLES} samples...")
    
    return samples


def measure_warm_import(import_statement: str) -> float:
    """Measure import time when already in memory (warm)."""
    # Clear any existing imports
    modules_to_clear = [
        'vllm.transformers_utils.configs',
        'vllm.transformers_utils.configs.afmoe',
        'vllm.transformers_utils.configs.chatglm',
        # Add more as needed
    ]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
    
    t0 = time.perf_counter()
    exec(import_statement)
    t1 = time.perf_counter()
    return t1 - t0


def measure_config_access(config_name: str) -> float:
    """Measure time to access a specific config (lazy loading overhead)."""
    # Ensure base module is imported
    import vllm.transformers_utils.configs as configs
    
    t0 = time.perf_counter()
    config_class = getattr(configs, config_name)
    t1 = time.perf_counter()
    return t1 - t0


def measure_vllm_startup(model: str = "Qwen/Qwen3-0.6B") -> List[float]:
    """Measure vLLM LLM() initialization time."""
    samples: List[float] = []
    
    cmd = [
        sys.executable,
        "-c",
        (
            "import time; "
            "from vllm import LLM; "
            f"t0 = time.perf_counter(); "
            f"llm = LLM(model='{model}', skip_tokenizer_init=True); "
            "t1 = time.perf_counter(); "
            "print(t1 - t0)"
        )
    ]
    
    print(f"Measuring vLLM startup with model: {model}")
    for i in range(5):  # Fewer samples as this is slower
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = float(result.stdout.strip())
        samples.append(elapsed)
        print(f"  Completed {i + 1}/5 samples...")
    
    return samples


def print_stats(label: str, samples: List[float]):
    """Print statistical summary."""
    print(f"\n{label}:")
    print(f"  Samples: {len(samples)}")
    print(f"  Mean:    {statistics.mean(samples):.6f} s")
    print(f"  Median:  {statistics.median(samples):.6f} s")
    print(f"  Min:     {min(samples):.6f} s")
    print(f"  Max:     {max(samples):.6f} s")
    if len(samples) > 1:
        print(f"  StdDev:  {statistics.stdev(samples):.6f} s")


def main():
    print("=" * 70)
    print("vLLM Configs Lazy Loading Benchmark")
    print("=" * 70)
    
    # 1. Measure base module import (should be much faster with lazy loading)
    print("\n[1] Base module import (cold)")
    lazy_samples = measure_cold_import("import vllm.transformers_utils.configs")
    print_stats("Lazy Loading (new)", lazy_samples)
    
    # 2. Measure accessing a specific config
    print("\n[2] Accessing specific config (DeepseekVLV2Config)")
    access_times = []
    for _ in range(100):  # Multiple accesses
        # Need to clear module between runs
        if 'vllm.transformers_utils.configs' in sys.modules:
            del sys.modules['vllm.transformers_utils.configs']
        if 'vllm.transformers_utils.configs.deepseek_vl2' in sys.modules:
            del sys.modules['vllm.transformers_utils.configs.deepseek_vl2']
        
        t = measure_config_access("DeepseekVLV2Config")
        access_times.append(t)
    
    print_stats("Config Access (lazy)", access_times)
    print(f"  Average overhead per access: {statistics.mean(access_times) * 1000:.3f} ms")
    
    # 3. Measure vLLM startup impact (optional, slower)
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("\n[3] vLLM LLM() initialization")
        startup_samples = measure_vllm_startup()
        print_stats("vLLM Startup", startup_samples)
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()