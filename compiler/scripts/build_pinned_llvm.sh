#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOTSTRAP_SCRIPT="${ROOT_DIR}/scripts/bootstrap_llvm.sh"
LLVM_PROJECT_DIR="${ROOT_DIR}/third_party/llvm-project"
LLVM_SRC_DIR="${LLVM_PROJECT_DIR}/llvm"
BUILD_DIR="${1:-${ROOT_DIR}/../build/llvm-project}"
INSTALL_DIR="${2:-${ROOT_DIR}/../build/llvm-install}"
JOBS="${JOBS:-$(nproc)}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"

if [[ ! -x "${BOOTSTRAP_SCRIPT}" ]]; then
  echo "error: bootstrap script not found at ${BOOTSTRAP_SCRIPT}" >&2
  exit 1
fi

"${BOOTSTRAP_SCRIPT}"

if [[ ! -f "${LLVM_SRC_DIR}/CMakeLists.txt" ]]; then
  echo "error: llvm source tree not found under ${LLVM_SRC_DIR}" >&2
  exit 1
fi

echo "configuring pinned llvm-project"
echo "  source:  ${LLVM_SRC_DIR}"
echo "  build:   ${BUILD_DIR}"
echo "  install: ${INSTALL_DIR}"

"${CMAKE_BIN}" -S "${LLVM_SRC_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

"${CMAKE_BIN}" --build "${BUILD_DIR}" --target install -j"${JOBS}"

cat <<EOF
done: pinned llvm-project built and installed locally

use it with:
  export LLVM_DIR=${INSTALL_DIR}/lib/cmake/llvm
  export MLIR_DIR=${INSTALL_DIR}/lib/cmake/mlir
  ./compiler/scripts/configure_compiler.sh
  cmake --build build/npu_compiler --target npu-opt npu-translate -j${JOBS}
EOF
