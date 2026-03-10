#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/../build/npu_compiler}"

# Preferred usage is to pass LLVM_DIR / MLIR_DIR explicitly. For local
# convenience, allow deriving them from a Merlin build root.
MERLIN_BUILD_DIR="${MERLIN_BUILD_DIR:-}"
if [[ -n "${MERLIN_BUILD_DIR}" ]]; then
  LLVM_DIR_DEFAULT="${MERLIN_BUILD_DIR}/llvm-project/lib/cmake/llvm"
  MLIR_DIR_DEFAULT="${MERLIN_BUILD_DIR}/lib/cmake/mlir"
else
  LLVM_DIR_DEFAULT=""
  MLIR_DIR_DEFAULT=""
fi

LLVM_DIR="${LLVM_DIR:-${LLVM_DIR_DEFAULT}}"
MLIR_DIR="${MLIR_DIR:-${MLIR_DIR_DEFAULT}}"

if [[ -z "${LLVM_DIR}" || -z "${MLIR_DIR}" ]]; then
  cat >&2 <<EOF
error: LLVM_DIR and MLIR_DIR must be set, or MERLIN_BUILD_DIR must point to the
Merlin build root that contains:
  llvm-project/lib/cmake/llvm
  lib/cmake/mlir
EOF
  exit 1
fi

if [[ ! -f "${LLVM_DIR}/LLVMConfig.cmake" ]]; then
  echo "error: LLVMConfig.cmake not found under ${LLVM_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MLIR_DIR}/MLIRConfig.cmake" ]]; then
  echo "error: MLIRConfig.cmake not found under ${MLIR_DIR}" >&2
  exit 1
fi

echo "configuring standalone NPU compiler in ${BUILD_DIR}"
echo "  LLVM_DIR=${LLVM_DIR}"
echo "  MLIR_DIR=${MLIR_DIR}"

/usr/bin/cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DMLIR_DIR="${MLIR_DIR}"

echo "done. build with:"
echo "  /usr/bin/cmake --build ${BUILD_DIR} --target npu-opt npu-translate -j\$(nproc)"
