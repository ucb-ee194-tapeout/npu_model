# from typing import Optional, List, TYPE_CHECKING

# from .exu import ExecutionUnit, InFlightInsn
# from .stage_data import StageData
# from ..trace.logger import Logger

# if TYPE_CHECKING:
#     from ..software.insn import Instruction


# class LoadStoreUnit(ExecutionUnit):
#     """
#     Load/Store Unit for DMA and memory operations.

#     Only allows 1 in-flight instruction at a time. When busy, it won't
#     claim new instructions, causing backpressure on the pipeline.
#     """

#     def __init__(
#         self,
#         name: str,
#         logger: Logger,
#         lane_id: int = 0,
#         dispatch_lane_id: int = 1,
#     ) -> None:
#         super().__init__(
#             name,
#             logger,
#             lane_id=lane_id,
#             dispatch_lane_id=dispatch_lane_id,
#         )
#         self.reset()

#     def reset(self) -> None:
#         self._in_flight: Optional[InFlightInsn] = None
#         self._completed_instructions: List["Instruction"] = []
#         self._pending_completion: Optional["Instruction"] = None
#         self._total_instructions = 0
#         self._busy_cycles = 0

#     def tick(self, diu_output: StageData[Optional["Instruction"]]) -> None:
#         """
#         Execute one cycle with single-issue constraint.

#         Only claims a new instruction if no instruction is currently in flight.
#         """
#         # Log deferred completion from last cycle
#         if self._pending_completion is not None:
#             self.logger.log_stage_end(
#                 self._pending_completion.id,
#                 "E",
#                 lane=self.lane_id,
#             )
#             self.logger.log_retire(self._pending_completion.id)
#             self._pending_completion = None

#         self._completed_instructions = []

#         # Only claim if we have no in-flight instruction
#         dispatched_insn = None
#         if self._in_flight is None:
#             dispatched_insn = diu_output.claim()

#         # Accept new instruction
#         if dispatched_insn is not None:
#             self._in_flight = InFlightInsn(
#                 insn=dispatched_insn,
#                 cycles_remaining=dispatched_insn.op.ex_delay
#             )
#             self._total_instructions += 1
#             # Log: end dispatch, start execute
#             self.logger.log_stage_end(
#                 dispatched_insn.id,
#                 "D",
#                 lane=self.dispatch_lane_id,
#             )
#             self.logger.log_stage_start(
#                 dispatched_insn.id,
#                 "E",
#                 lane=self.lane_id,
#             )

#         # Track if EXU was busy
#         if self._in_flight is not None:
#             self._busy_cycles += 1

#         # Process in-flight instruction
#         if self._in_flight is not None:
#             self._in_flight.cycles_remaining -= 1
#             if self._in_flight.cycles_remaining <= 0:
#                 self._completed_instructions.append(self._in_flight.insn)
#                 self._pending_completion = self._in_flight.insn
#                 self._in_flight = None

#     def flush_completions(self) -> None:
#         """Flush any pending completions (call at end of simulation)."""
#         if self._pending_completion is not None:
#             self.logger.log_stage_end(
#                 self._pending_completion.id,
#                 "E",
#                 lane=self.lane_id,
#             )
#             self.logger.log_retire(self._pending_completion.id)
#             self._pending_completion = None

#     def has_in_flight(self) -> bool:
#         """Check if there is an in-flight instruction."""
#         return self._in_flight is not None

#     @property
#     def completed_instructions(self) -> List["Instruction"]:
#         """Instructions completed this cycle."""
#         return self._completed_instructions

#     @property
#     def total_instructions(self) -> int:
#         """Total instructions executed."""
#         return self._total_instructions

#     @property
#     def busy_cycles(self) -> int:
#         """Number of cycles the EXU was busy."""
#         return self._busy_cycles


# class WeightLoadStoreUnit(LoadStoreUnit):
#     """Load/Store Unit for weight memory operations."""

#     def __init__(
#         self,
#         name: str,
#         logger: Logger,
#         lane_id: int = 0,
#         dispatch_lane_id: int = 1,
#     ) -> None:
#         super().__init__(
#             name,
#             logger,
#             lane_id=lane_id,
#             dispatch_lane_id=dispatch_lane_id,
#         )


# class MatrixLoadStoreUnit(LoadStoreUnit):
#     """Load/Store Unit for matrix/activation memory operations."""

#     def __init__(
#         self,
#         name: str,
#         logger: Logger,
#         lane_id: int = 0,
#         dispatch_lane_id: int = 1,
#     ) -> None:
#         super().__init__(
#             name,
#             logger,
#             lane_id=lane_id,
#             dispatch_lane_id=dispatch_lane_id,
#         )
