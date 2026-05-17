#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/build"

clang++ -std=c++17 \
  "$ROOT/cpp/llama_branch_worker.cpp" \
  -o "$ROOT/build/llama_branch_worker" \
  -I/opt/homebrew/include \
  -L/opt/homebrew/lib \
  -lllama \
  -Wl,-rpath,/opt/homebrew/lib

echo "$ROOT/build/llama_branch_worker"
