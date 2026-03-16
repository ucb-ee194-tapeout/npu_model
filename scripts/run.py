#!/usr/bin/env python3
"""
NPU Performance Model - Simulation Runner

Usage:
    uv run scripts/run.py [options]

Options:
    -p, --program       Program to run (default: AddiProgram)
    --hardware_config   Hardware configuration (default: DefaultHardwareConfig)
    -o, --output        Output trace file
    --max-cycles        Maximum cycles to simulate

Features:
    - Runs cycle-accurate NPU simulation
    - Generates Perfetto trace for visualization
    - Reports performance metrics (IPC, utilization, etc.)
    - Checks correctness against golden results (if defined)
"""

import argparse

import torch

import npu_model
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation

from npu_model.configs.programs import *  # noqa: F401, F403
from npu_model.configs.hardware import *  # noqa: F401, F403
from npu_model.configs.isa_definition import *  # noqa: F401, F403


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NPU Performance Model Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run.py
    python scripts/run.py -o my_trace.json
    python scripts/run.py --max-cycles 5000
        """,
    )
    parser.add_argument(
        "--hardware_config",
        type=str,
        default="DefaultHardwareConfig",
        help="Hardware configuration",
    )
    parser.add_argument(
        "-p",
        "--program",
        type=str,
        default="AddiProgram",
        help="Program to run",
    )
    parser.add_argument(
        "-o", "--output", default="trace.json", help="Output trace file"
    )
    parser.add_argument(
        "--max-cycles", type=int, default=10000, help="Maximum cycles to simulate"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NPU Performance Model - Simulation")
    print("=" * 60)

    # Create simulation environment
    try:
        hardware_config = eval(args.hardware_config)()
    except NameError:
        print(f"Hardware config '{args.hardware_config}' not found.")
        print("available options are:")
        print(f"  {', '.join(npu_model.configs.hardware.__all__)}")
        return
    try:
        program = eval(args.program)()
    except NameError:
        print(f"Program '{args.program}' not found.")
        print("available options are:")
        print(f"  {', '.join(npu_model.configs.programs.__all__)}")
        return
    sim = Simulation(
        hardware_config=hardware_config,
        logger_config=LoggerConfig(filename=args.output),
        program=program,
    )
    sim.run(max_cycles=args.max_cycles)

    # Check correctness against golden result
    if hasattr(program, "golden_result") and program.golden_result:
        print("\n" + "=" * 60)
        print("Correctness Check")
        print("=" * 60)

        output_base, golden_tensor = program.golden_result
        size = golden_tensor.numel() * golden_tensor.element_size()

        try:
            # Read actual output from memory
            mem_data = sim.core.arch_state.read_memory(output_base, size)
            actual = mem_data.view(golden_tensor.dtype).reshape(golden_tensor.shape).clone()

            # Compare with golden result
            if torch.allclose(actual.float(), golden_tensor.float(), rtol=1e-2, atol=1e-2):
                max_diff = (actual.float() - golden_tensor.float()).abs().max()
                print(f"✓ PASS - Output matches golden result!")
                print(f"  Max difference: {max_diff:.6e}")
            else:
                max_diff = (actual.float() - golden_tensor.float()).abs().max()
                mean_diff = (actual.float() - golden_tensor.float()).abs().mean()
                print(f"✗ FAIL - Output does NOT match golden result!")
                print(f"  Max difference:  {max_diff:.6e}")
                print(f"  Mean difference: {mean_diff:.6e}")
                print(f"  Tolerance: rtol=1e-2, atol=1e-2")

            # Print sample of actual output
            print(f"\nActual output (from address {hex(output_base)}):")
            print(f"  Shape: {actual.shape}, dtype: {actual.dtype}")
            print(f"  Sample (first 32 elements): {actual.flatten()[:32]}")

            print(f"\nExpected output:")
            print(f"  Shape: {golden_tensor.shape}, dtype: {golden_tensor.dtype}")
            print(f"  Sample (first 32 elements): {golden_tensor.flatten()[:32]}")

        except Exception as e:
            print(f"✗ ERROR during correctness check: {e}")
    else:
        print("\n" + "=" * 60)
        print("Note: No golden result defined for this program")
        print("=" * 60)

        # Try to print memory from common output locations
        common_addrs = [0x1800, 0x3000]
        for addr in common_addrs:
            try:
                data = sim.core.arch_state.read_memory(addr, 64 * 16 * 2).view(torch.bfloat16)
                if data.abs().sum() > 0:  # If non-zero
                    print(f"\nMemory at {hex(addr)} (first 32 elements):")
                    print(data[:32])
            except Exception:
                pass


if __name__ == "__main__":
    main()
