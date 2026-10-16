#!/usr/bin/env python3
"""Export compiler-facing true assembly syntax/encoding metadata as JSON."""

import argparse
import json
from pathlib import Path

from npu_model.isa_metadata import export_isa


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Atlas ISA metadata as JSON")
    parser.add_argument("-o", "--output", type=Path, help="write JSON to this path (default: stdout)")
    args = parser.parse_args()
    rendered = json.dumps(export_isa(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
