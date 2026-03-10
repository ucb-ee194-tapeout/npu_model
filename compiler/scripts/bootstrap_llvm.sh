#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN_FILE="${ROOT_DIR}/third_party/llvm-project.pin.json"
DST_DIR="${ROOT_DIR}/third_party/llvm-project"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to read ${PIN_FILE}" >&2
  exit 1
fi

REPO_URL="$(jq -r '.source.repository' "${PIN_FILE}")"
PIN_SHA="$(jq -r '.pin.llvm_project_commit' "${PIN_FILE}")"

if [[ -d "${DST_DIR}/.git" ]]; then
  echo "info: existing llvm-project checkout found at ${DST_DIR}"
else
  echo "info: cloning ${REPO_URL} into ${DST_DIR}"
  git clone "${REPO_URL}" "${DST_DIR}"
fi

pushd "${DST_DIR}" >/dev/null

echo "info: fetching and checking out ${PIN_SHA}"
git fetch --all --tags
git checkout "${PIN_SHA}"

popd >/dev/null

echo "done: llvm-project pinned at ${PIN_SHA}"
