#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "warning: compiler/scripts/configure_with_merlin.sh is deprecated; use compiler/scripts/configure_compiler.sh instead." >&2
exec "${SCRIPT_DIR}/configure_compiler.sh" "$@"
