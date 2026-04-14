from dataclasses import dataclass
import sys
from npu_model.hardware.config import HardwareConfig
from npu_model.logging import LoggerConfig, Logger
from npu_model.hardware import Core
from npu_model.software import Program

@dataclass
class ExecutionUnitStatistics:
    instructions: int
    busy_cycles: int
    utilization: float

@dataclass
class SimulationStatistics:
    cycles: int
    total_instructions: int
    ipc: float
    runtime_errors: int
    exu_stats: dict[str, ExecutionUnitStatistics]


class Simulation:
    def __init__(
        self,
        hardware_config: HardwareConfig,
        logger_config: LoggerConfig,
        program: Program,
        verbose: bool = True,
        ignore_runtime_errors: bool = False,
    ):
        """
        Create a simple NPU hardware configuration.

        Args:
            program: The program to execute
            logger: Trace logger for output
            config: Hardware configuration

        Returns:
            Configured Core ready to run
        """
        self.logger_config = logger_config
        self.hardware_config = hardware_config
        self.program = program
        self.verbose = verbose
        self.ignore_runtime_errors = ignore_runtime_errors
        self.runtime_errors: list[tuple[int, str, str]] = []

        # Create logger for trace output
        lane_names = {0: "IFU", 1: "DIU"}
        for idx, exu_name in enumerate(hardware_config.execution_units.keys()):
            lane_names[2 + idx] = exu_name
        self.logger = Logger(logger_config, lane_names=lane_names)

        isa = self.hardware_config.isa
        if self.verbose:
            print(f"\nISA loaded with {len(isa.operations)} operations")

        # Create core
        self.core = Core(
            config=hardware_config,
            logger=self.logger,
        )
        self.core.ignore_runtime_errors = ignore_runtime_errors
        self.core.runtime_error_reporter = self._report_runtime_error

        self.core.load_program(self.program)
        if self.verbose:
            print(f"Program loaded with {len(self.program)} instructions")

            print("\nHardware configured:")
            print("  - Fetch width: 1 instruction/cycle (in-order)")
            print(f"  - Execution units: {[str(exu) for exu in self.core.exus]}")
            print(f"  - Trace output: {self.logger_config.filename}")
            print(f"  - Bypass runtime errors: {self.ignore_runtime_errors}")

            # Run simulation
            print("\n" + "-" * 60)
            print("Running simulation...")
            print("-" * 60)

    def run(self, max_cycles: int = 10000):
        """Run simulation until completion or max_cycles."""

        # self.core.run(max_cycles=max_cycles)
        self.core.reset()

        self.cycle_count = 0

        while not self.core.is_finished() and self.cycle_count < max_cycles:
            self.core.tick()
            self.cycle_count += 1

        # Flush any pending completions in EXUs
        self.core.stop()

        # Close logger
        self.logger.close()

        # Get and print results
        stats = self.get_stats()

        if self.verbose:
            print("\nSimulation Complete!")
            print(f"\n{'Metric':<30} {'Value':>15}")
            print("-" * 45)
            print(f"{'Total Cycles':<30} {stats.cycles:>15}")
            print(f"{'Instructions Completed':<30} {stats.total_instructions:>15}")
            print(f"{'IPC (Instr per Cycle)':<30} {stats.ipc:>15.3f}")
            print(f"{'Suppressed Runtime Errors':<30} {stats.runtime_errors:>15}")

            print("\nExecution Unit Utilization")
            print("-" * 45)
            for exu_name, exu_stats in stats.exu_stats.items():
                print(f"  {exu_name}:")
                print(f"    Instructions: {exu_stats.instructions}")
                print(f"    Busy Cycles:  {exu_stats.busy_cycles}")
                print(f"    Utilization:  {exu_stats.utilization:.1%}")

            print("\nFinal register contents")
            print(f"XRF: {self.core.arch_state.xrf}")
            print(f"MRF[0]: {self.core.arch_state.mrf[0]}")
            print(f"MRF[1]: {self.core.arch_state.mrf[1]}")

            print("\n" + "=" * 60)
            print(f"\nTrace written to: {self.logger_config.filename}")
            print("Open with Perfetto (https://ui.perfetto.dev)")

    def get_stats(self) -> SimulationStatistics:
        """Get execution statistics."""
        stats = SimulationStatistics(
            cycles=self.cycle_count,
            total_instructions=self.core.total_completed,
            ipc=(
                self.core.total_completed / self.cycle_count
                if self.cycle_count > 0
                else 0.0
            ),
            runtime_errors=len(self.runtime_errors),
            exu_stats={}
        )

        for exu in self.core.exus:
            stats.exu_stats[exu.name] = ExecutionUnitStatistics(
                instructions= exu.total_instructions,
                busy_cycles=exu.busy_cycles,
                utilization=(
                    exu.busy_cycles / self.cycle_count if self.cycle_count > 0 else 0.0
                ),
            )

        return stats

    def _report_runtime_error(self, stage: str, exc: Exception) -> None:
        cycle = self.cycle_count + 1
        message = str(exc)
        self.runtime_errors.append((cycle, stage, message))
        warning = (
            f"\033[1;91mWARNING: bypassed runtime error at cycle {cycle} "
            f"in {stage}: {message}\033[0m"
        )
        print(warning, file=sys.stderr)
