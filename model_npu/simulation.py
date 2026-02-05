from model_npu.hardware.config import HardwareConfig
from model_npu.logging import LoggerConfig, Logger
from model_npu.hardware import Core
from model_npu.software import Program


class Simulation:
    def __init__(
        self,
        hardware_config: HardwareConfig,
        logger_config: LoggerConfig,
        program: Program,
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

        # Create logger for trace output
        lane_names = {0: "IFU", 1: "DIU"}
        for idx, exu_name in enumerate(hardware_config.execution_units.keys()):
            lane_names[2 + idx] = exu_name
        self.logger = Logger(logger_config, lane_names=lane_names)

        isa = self.hardware_config.isa
        print(f"\nISA loaded with {len(isa.operations)} operations")

        # Create core
        self.core = Core(
            config=hardware_config,
            logger=self.logger,
        )

        self.core.load_program(self.program)
        print(f"Program loaded with {len(self.program)} instructions")

        print("\nHardware configured:")
        print("  - Fetch width: 1 instruction/cycle (in-order)")
        print(f"  - Execution units: {[str(exu) for exu in self.core.exus]}")
        print(f"  - Trace output: {self.logger_config.filename}")

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

        print("\nSimulation Complete!")
        print(f"\n{'Metric':<30} {'Value':>15}")
        print("-" * 45)
        print(f"{'Total Cycles':<30} {stats['cycles']:>15}")
        print(f"{'Instructions Completed':<30} {stats['total_instructions']:>15}")
        print(f"{'IPC (Instr per Cycle)':<30} {stats['ipc']:>15.3f}")

        print("\nExecution Unit Utilization")
        print("-" * 45)
        for exu_name, exu_stats in stats["exu_stats"].items():
            print(f"  {exu_name}:")
            print(f"    Instructions: {exu_stats['instructions']}")
            print(f"    Busy Cycles:  {exu_stats['busy_cycles']}")
            print(f"    Utilization:  {exu_stats['utilization']:.1%}")

        print("\nFinal register contents")
        print(f"XRF: {self.core.arch_state.xrf}")
        print(f"MRF[0]: {self.core.arch_state.mrf[0]}")
        print(f"MRF[1]: {self.core.arch_state.mrf[1]}")

        print("\n" + "=" * 60)
        print(f"\nTrace written to: {self.logger_config.filename}")
        print("Open with Perfetto (https://ui.perfetto.dev)")

    def get_stats(self) -> dict:
        """Get execution statistics."""
        stats = {
            "cycles": self.cycle_count,
            "total_instructions": self.core.total_completed,
            "ipc": (
                self.core.total_completed / self.cycle_count
                if self.cycle_count > 0
                else 0
            ),
            "exu_stats": {},
        }

        for exu in self.core.exus:
            stats["exu_stats"][exu.name] = {
                "instructions": exu.total_instructions,
                "busy_cycles": exu.busy_cycles,
                "utilization": (
                    exu.busy_cycles / self.cycle_count if self.cycle_count > 0 else 0
                ),
            }

        return stats
