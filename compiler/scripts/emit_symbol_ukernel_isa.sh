#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/../build/npu_compiler}"
SYMBOL="${2:-}"
OUT_FILE="${3:-${ROOT_DIR}/test/outputs/npu_symbol_ukernel_isa.txt}"

if [[ -z "${SYMBOL}" ]]; then
  echo "usage: $0 <build-dir> <ukernel-symbol> [out-file]" >&2
  exit 1
fi

NPU_OPT="${BUILD_DIR}/tools/npu-opt"
NPU_TRANSLATE="${BUILD_DIR}/tools/npu-translate"

if [[ ! -x "${NPU_OPT}" ]]; then
  echo "error: npu-opt not found at ${NPU_OPT}" >&2
  exit 1
fi
if [[ ! -x "${NPU_TRANSLATE}" ]]; then
  echo "error: npu-translate not found at ${NPU_TRANSLATE}" >&2
  exit 1
fi

LHS_TYPE=""
RHS_TYPE=""
OUT_TYPE="tensor<64x16xf32>"

case "${SYMBOL}" in
  npu_uk_matmul_*|npu_uk_gemma_mlp_*)
    LHS_TYPE="tensor<64x32xf8E4M3FN>"
    RHS_TYPE="tensor<32x16xf8E4M3FN>"
    ;;
  npu_uk_gemma_attention_*)
    LHS_TYPE="tensor<64x16xf8E4M3FN>"
    RHS_TYPE="tensor<16x16xf8E4M3FN>"
    ;;
  *)
    echo "error: unsupported symbol '${SYMBOL}'" >&2
    echo "supported prefixes: npu_uk_matmul_, npu_uk_gemma_mlp_, npu_uk_gemma_attention_" >&2
    exit 1
    ;;
esac

TMP_INPUT="$(mktemp /tmp/npu_symbol_ukernel_XXXXXX.mlir)"
trap 'rm -f "${TMP_INPUT}"' EXIT

cat > "${TMP_INPUT}" <<EOF
module {
  func.func @entry(%lhs: ${LHS_TYPE}, %rhs: ${RHS_TYPE}) -> ${OUT_TYPE} {
    %0 = npu_kernel.ukernel_generic "${SYMBOL}"(%lhs, %rhs) : ${LHS_TYPE}, ${RHS_TYPE} -> ${OUT_TYPE}
    return %0 : ${OUT_TYPE}
  }
}
EOF

"${NPU_OPT}" "${TMP_INPUT}" \
  -convert-npu-kernel-to-schedule \
  -convert-npu-schedule-to-isa \
  | "${NPU_TRANSLATE}" --mlir-to-npu-text-isa > "${OUT_FILE}"

echo "wrote ${OUT_FILE}"
