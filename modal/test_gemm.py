"""Run INT8 GEMM CUDA tests on a Modal GPU.

Setup (once):
  pip install -r requirements-modal.txt
  modal setup

Run:
  modal run modal/test_gemm.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

APP_NAME = "quantized-gemm-test"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("build-essential")
    .add_local_dir(
        local_path=str(PROJECT_ROOT / "src"),
        remote_path="/root/project/src",
    )
)

app = modal.App(APP_NAME)


@app.function(gpu="T4", image=image, timeout=600)
def run_int8_gemm_tests() -> int:
    project = Path("/root/project")

    compile_cmd = [
        "nvcc",
        "-std=c++17",
        "-O2",
        "-DRUN_INT8_GEMM_TEST",
        "-Isrc/reference",
        "-Isrc/kernels",
        "src/kernels/gemm.cu",
        "src/reference/gemm.cpp",
        "-o",
        "test_int8_gemm",
    ]

    print("=== Compiling ===")
    compile = subprocess.run(
        compile_cmd,
        cwd=project,
        capture_output=True,
        text=True,
    )
    print(compile.stdout)
    if compile.stderr:
        print(compile.stderr)
    if compile.returncode != 0:
        print(f"Compile failed with exit code {compile.returncode}")
        return compile.returncode

    print("\n=== Running tests ===")
    run = subprocess.run(
        ["./test_int8_gemm"],
        cwd=project,
        capture_output=True,
        text=True,
    )
    print(run.stdout)
    if run.stderr:
        print(run.stderr)

    return run.returncode


@app.local_entrypoint()
def main() -> None:
    exit_code = run_int8_gemm_tests.remote()
    if exit_code != 0:
        raise SystemExit(exit_code)
