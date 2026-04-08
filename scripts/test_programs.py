#!/usr/bin/env python3
"""
Test execution of all registered programs.

Runs each program from npu_model.configs.programs with the default hardware
config and a bounded cycle count. Suitable for local runs and CI (e.g. GitHub Actions).

Usage:
    uv run python scripts/test_programs.py
    uv run python scripts/test_programs.py --max-cycles 5000
"""

import argparse
import sys
import tempfile
from pathlib import Path

# Add project root for imports when run as script
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import npu_model
import torch
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation

# Register all programs and hardware configs (same as run.py)
from npu_model.configs.programs import *  # noqa: F401, F403
from npu_model.configs.hardware import *  # noqa: F401, F403
from npu_model.configs.isa_definition import *  # noqa: F401, F403


def main():
    parser = argparse.ArgumentParser(
        description="Run all programs to verify execution."
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=10000,
        help="Maximum cycles per simulation (default: 10000)",
    )
    parser.add_argument(
        "--hardware_config",
        type=str,
        default="DefaultHardwareConfig",
        help="Hardware configuration class name",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    program_names = getattr(npu_model.configs.programs, "__all__", [])
    if not program_names:
        print("No programs found in npu_model.configs.programs", file=sys.stderr)
        sys.exit(1)

    try:
        hardware_config_cls = getattr(npu_model.configs.hardware, args.hardware_config)
    except AttributeError:
        print(
            f"Hardware config '{args.hardware_config}' not found.",
            file=sys.stderr,
        )
        print(
            "Available:",
            ", ".join(getattr(npu_model.configs.hardware, "__all__", [])),
            file=sys.stderr,
        )
        sys.exit(1)

    failed = []
    for name in sorted(program_names):
        try:
            program_cls = getattr(npu_model.configs.programs, name)
            program = program_cls()
        except Exception as e:
            failed.append((name, e))
            if args.verbose:
                print(f"FAIL {name}: instantiate: {e}", file=sys.stderr)
            continue

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            trace_path = f.name
        try:
            sim = Simulation(
                hardware_config=hardware_config_cls(),
                logger_config=LoggerConfig(filename=trace_path),
                program=program,
                verbose=args.verbose,
            )
            if not args.verbose:
                # Redirect stdout so we only see summary
                import io

                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
            sim.run(max_cycles=args.max_cycles)

            # Verify golden result if program defines it
            if hasattr(program, "golden_result") and program.golden_result:
                output_base, golden_tensor = program.golden_result
                size = golden_tensor.numel() * golden_tensor.element_size()
                mem_data = sim.core.arch_state.read_dram(output_base, size)
                actual = (
                    mem_data.view(golden_tensor.dtype)
                    .reshape(golden_tensor.shape)
                    .clone()
                )
                if not torch.allclose(
                    actual.float(), golden_tensor.float(), rtol=1e-2, atol=1e-2
                ):
                    diff = (actual.float() - golden_tensor.float()).abs().max()
                    raise AssertionError(
                        f"Golden check failed: max diff = {diff:.6f}. \nResult = {actual.float()}\n Expected = {golden_tensor}"
                    )

            if not args.verbose:
                sys.stdout = old_stdout
            try:
                Path(trace_path).unlink(missing_ok=True)
            except (PermissionError, OSError):
                pass
        except Exception as e:
            if not args.verbose:
                try:
                    sys.stdout = old_stdout
                except NameError:
                    pass
            failed.append((name, e))
            if args.verbose:
                print(f"FAIL {name}: {e}", file=sys.stderr)
        else:
            if args.verbose:
                print(f"OK   {name}")
        finally:
            try:
                Path(trace_path).unlink(missing_ok=True)
            except (PermissionError, OSError):
                pass

    if failed:
        print(
            f"\n{len(failed)} of {len(program_names)} program(s) failed.",
            file=sys.stderr,
        )
        for name, err in failed:
            print(f"  {name}: {err}", file=sys.stderr)
        sys.exit(1)
    print(f"All {len(program_names)} program(s) passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
