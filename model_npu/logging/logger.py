from dataclasses import dataclass
from enum import Enum

import json
from typing import Dict, Optional, Tuple, Any


@dataclass
class LaneType(Enum):
    IFU = 0
    DIU = 1
    EXU_BASE = 2


@dataclass
class RetireType(Enum):
    """Instruction retirement type."""

    RETIRE = 0
    FLUSH = 1


@dataclass
class LoggerConfig:
    filename: str = "trace.json"


class Logger:
    """
    Perfetto (Chrome Trace Event) Logger.

    - Each lane becomes a separate track (thread).
    - Stage start/end is recorded as a complete event ("X") over time.
    - log_cycle advances the timestamp counter.
    """

    FU_PID = 0
    ARCH_PID = 1

    def __init__(
        self,
        config: LoggerConfig,
        process_name: str = "NPU",
        lane_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.config = config
        self.file = open(config.filename, "w")
        self.first_event = True
        self.insn_labels: Dict[int, str] = {}
        self.active: Dict[Tuple[int, str, int], int] = {}
        self.lane_names = lane_names or {}
        self.arch_threads: Dict[Tuple[str, int], Tuple[int, str]] = {}
        self.ts = 1

        self.file.write("[")
        self._write_event(
            {
                "name": "process_name",
                "ph": "M",
                "pid": Logger.FU_PID,
                "tid": 0,
                "args": {"name": process_name},
            }
        )
        for lane in sorted(self.lane_names):
            self._write_event(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": Logger.FU_PID,
                    "tid": lane,
                    "args": {"name": self.lane_names[lane]},
                }
            )
        self._write_event(
            {
                "name": "process_name",
                "ph": "M",
                "pid": Logger.ARCH_PID,
                "tid": 0,
                "args": {"name": "ArchState"},
            }
        )

    def close(self) -> None:
        """Close the trace file."""
        self.file.write("]\n")
        self.file.close()

    def _write_event(self, event: Dict[str, Any]) -> None:
        if not self.first_event:
            self.file.write(",\n")
        else:
            self.first_event = False
        self.file.write(json.dumps(event, separators=(",", ":")))

    def log_cycle(self, elapsed: int) -> None:
        self.ts += elapsed

    def log_insn(self, insn_id: int, label: str, thread_id: int = 0) -> None:
        """Record instruction label for later display."""
        self.insn_labels[insn_id] = f"{insn_id}: {label}"

    def log_retire(
        self, insn_id: int, retire_type: RetireType = RetireType.RETIRE
    ) -> None:
        pass

    def log_stage_start(
        self, insn_id: int, stage: str, lane: int = 0, cycle: int = 0
    ) -> None:
        """Mark the start of a stage on the given lane."""
        key = (insn_id, stage, lane)
        if key in self.active:
            return
        self.active[key] = cycle

    def log_stage_end(
        self, insn_id: int, stage: str, lane: int = 0, cycle: int = 0
    ) -> None:
        """Mark the end of a stage on the given lane."""
        key = (insn_id, stage, lane)
        if key not in self.active:
            return
        start_ts = self.active.pop(key)
        dur = cycle - start_ts
        if dur < 0:
            dur = 0
        label = self.insn_labels.get(insn_id, f"insn-{insn_id}")
        self._write_event(
            {
                "name": label,
                "cat": stage,
                "ph": "X",
                "pid": Logger.FU_PID,
                "tid": lane,
                "ts": start_ts,
                "dur": dur,
                "args": {"insn_id": insn_id, "stage": stage},
            }
        )

    def log_dependency(
        self, consumer_insn_id: int, producer_insn_id: int, dep_type: int = 0
    ) -> None:
        """Optional dependency logging (currently no-op)."""
        return

    def log_arch_value(self, regfile: str, index: int, value: int) -> None:
        # print(f"Logging architectural state value: {regfile}[{index}] = {value}")
        """Log architectural state value changes as counter events."""
        key = (regfile, index)
        if key not in self.arch_threads:
            if regfile == "xrf":
                tid = index
                name = f"{regfile}[{index:02d}]"
            elif regfile == "pc":
                tid = 1000
                name = "pc"
            else:
                tid = 2000 + index
                name = f"{regfile}[{index:02d}]"
            self.arch_threads[key] = (tid, name)
            self._write_event(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": Logger.ARCH_PID,
                    "tid": tid,
                    "args": {"name": name},
                }
            )
        tid, name = self.arch_threads[key]
        self._write_event(
            {
                "name": name,
                "ph": "C",
                "pid": Logger.ARCH_PID,
                "tid": tid,
                "ts": self.ts,
                "args": {"value": value},
            }
        )
