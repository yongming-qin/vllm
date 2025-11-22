from __future__ import annotations
import subprocess
import statistics
from typing import List

N = 10  # number of samples


# "import vllm.transformers_utils.configs; "
# "from vllm.transformers_utils.configs import DeepseekV3Config; "

cmd = [
        "python",
        "-c",
        (
            "import time; "
            "t0 = time.perf_counter(); "
            "import vllm.transformers_utils.configs; "
            "t1 = time.perf_counter(); "
            "print(t1 - t0)"
        )
    ]

def measure_once() -> float:
    # Run a Python one-liner in a fresh process
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def main() -> None:
    samples: List[float] = []
    
    print(f"running {N} times of command: {cmd}")

    for _ in range(N):
        samples.append(measure_once())

    
    print(f"Runs: {N}")
    print(f"Mean:  {statistics.mean(samples):.6f} s")
    print(f"Median:{statistics.median(samples):.6f} s")
    print(f"Min:   {min(samples):.6f} s")
    print(f"Max:   {max(samples):.6f} s")


if __name__ == "__main__":
    main()
