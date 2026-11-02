#!/usr/bin/env python3
"""
benchmark.py

Replace ``outputBufferSize`` in ``src/main.cpp`` with each value in
``message_sizes``. After each update, run ``./tacos.sh build`` and
``./tacos.sh run``, keep only the final numeric line of output, and append it
to ``results.txt``. Restore ``main.cpp`` to its original contents at the end.
"""

import os
import sys
import re
import subprocess

file_path = os.path.dirname(os.path.realpath(__file__))

main_cpp_path = os.path.join(file_path, 'src', 'main.cpp')
results_txt_path = os.path.join(file_path, 'results.txt')

message_sizes = [
    1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
    1024 * 1024 * 1024,
    4 * 1024 * 1024 * 1024,
    16 * 1024 * 1024 * 1024
]

def load_original():
    """Read and return the original ``main.cpp`` contents."""
    if not os.path.exists(main_cpp_path):
        print(f"Error: {main_cpp_path} was not found", file=sys.stderr)
        sys.exit(1)
    with open(main_cpp_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_content(content):
    """Write ``content`` back to ``main.cpp``."""
    with open(main_cpp_path, 'w', encoding='utf-8') as f:
        f.write(content)

def replace_size(content: str, size: int) -> str:
    """Use regex to replace the ``outputBufferSize`` value with ``size``."""
    pattern = re.compile(
        r'(const\s+Collective::ChunkSize\s+outputBufferSize\s*=\s*)\d+(\s*;)',
        re.MULTILINE
    )
    return pattern.sub(lambda m: f"{m.group(1)}{size}{m.group(2)}", content)

def build_and_run() -> str:
    """Build and run the benchmark, then return the final numeric line."""
    subprocess.run(['./tacos.sh', 'build'], check=True)
    proc = subprocess.run(['./tacos.sh', 'run'], capture_output=True, text=True, check=True)
    lines = proc.stdout.strip().splitlines()
    return lines[-1].strip()

def main():
    original = load_original()

    open(results_txt_path, 'w', encoding='utf-8').close()

    try:
        for size in message_sizes:
            print(f">>> Testing size: {size}")
            modified = replace_size(original, size)
            write_content(modified)

            result = build_and_run()
            print(f"    -> {result}")

            with open(results_txt_path, 'a', encoding='utf-8') as f:
                f.write(f"{result}\n")

    except subprocess.CalledProcessError as e:
        print(f"[Error] External command failed: {e}", file=sys.stderr)

    finally:
        print(">>> Restoring original main.cpp")
        write_content(original)

    print(f">>> Done. Results saved to {results_txt_path}")

if __name__ == '__main__':
    main()
