#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/../build/npu_compiler}"
INPUT_FILE="${2:-${ROOT_DIR}/test/lowering/linalg_generic_fp8_ukernel_pipeline.mlir}"
OUT_DIR="${3:-${ROOT_DIR}/test/outputs/ukernel_pipeline}"

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
if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "error: input file not found at ${INPUT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "== stage 1: linalg -> npu_kernel =="
"${NPU_OPT}" "${INPUT_FILE}" \
  -convert-linalg-to-npu-kernel \
  | tee "${OUT_DIR}/01_kernel.mlir"

echo "== stage 2: kernel -> schedule =="
"${NPU_OPT}" "${INPUT_FILE}" \
  -convert-linalg-to-npu-kernel \
  -convert-npu-kernel-to-schedule \
  | tee "${OUT_DIR}/02_schedule.mlir"

echo "== stage 3: schedule -> isa =="
"${NPU_OPT}" "${INPUT_FILE}" \
  -convert-linalg-to-npu-kernel \
  -convert-npu-kernel-to-schedule \
  -convert-npu-schedule-to-isa \
  | tee "${OUT_DIR}/03_isa.mlir"

echo "== stage 4: textual isa =="
"${NPU_OPT}" "${INPUT_FILE}" \
  -convert-linalg-to-npu-kernel \
  -convert-npu-kernel-to-schedule \
  -convert-npu-schedule-to-isa \
  | "${NPU_TRANSLATE}" --mlir-to-npu-text-isa \
  | tee "${OUT_DIR}/04_isa.txt"

echo "== full pass trace with --mlir-print-ir-after-all =="
"${NPU_OPT}" "${INPUT_FILE}" \
  -convert-linalg-to-npu-kernel \
  -convert-npu-kernel-to-schedule \
  -convert-npu-schedule-to-isa \
  --mlir-disable-threading \
  --mlir-print-ir-after-all \
  --mlir-print-ir-module-scope \
  > "${OUT_DIR}/trace_after_all.log" 2>&1

echo "wrote:"
echo "  ${OUT_DIR}/01_kernel.mlir"
echo "  ${OUT_DIR}/02_schedule.mlir"
echo "  ${OUT_DIR}/03_isa.mlir"
echo "  ${OUT_DIR}/04_isa.txt"
echo "  ${OUT_DIR}/trace_after_all.log"
