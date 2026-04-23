#!/usr/bin/env bash
set -euo pipefail

ASM_DIR="npu_model/configs/programs/asm"
BIN_DIR="npu_model/configs/programs/bin"
HEX_DIR="npu_model/configs/programs/hex"

for f in "$ASM_DIR"/*.S; do
    name=$(basename "$f" .S)
    uv run scripts/assemble.py -p "$f" \
        --out-bin "$BIN_DIR/$name.bin" \
        --out-hex "$HEX_DIR/$name.hex"
done
