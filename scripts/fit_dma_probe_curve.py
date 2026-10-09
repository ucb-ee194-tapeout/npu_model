"""Turn DMA probe logs into a Mess bandwidth-latency curve file.

Reads the VCS (or board) logs of the programs ``scripts/gen_dma_probe_programs.py``
writes, decodes CSR_DBG1 (probe cycles in the low 16 bits, traffic-phase
cycles in the high 16 bits), computes the delivered background bandwidth of
each program and writes the points grouped by background read percentage in
the Mess curve format, with the Atlas extensions ``accessBytes`` and
``frequencyGHz``.

    python scripts/fit_dma_probe_curve.py --logs .../atlas_perf_dma_probe_*.log \\
        --out npu_model/configs/memory_curves/ee290sim_vcs_probe.json

The probe's cycle count is the lone A-to-D round trip plus the engine's
command overhead; ``--overhead`` (default 5: the engine's cycles from the
timed window start to the A fire and from the D fire to retirement, as the
model's DMA engine spends them) is subtracted so the latencies in the file
are memory latencies as the backend applies them.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

NAME = re.compile(r"perf_dma_probe_(\d+)r_(\d+)bg_(\d+)(_st)?")
DBG1 = re.compile(r"dbg1_cycles\s*=\s*(\d+)")
BG_BYTES = re.compile(r"background bytes = (\d+)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--asm-dir", type=Path, default=Path("../baremetal/assembly"),
                        help="Where the generated .S files are, to read the background byte count")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frequency-ghz", type=float, default=0.5)
    parser.add_argument("--beat-bytes", type=int, default=32)
    parser.add_argument("--overhead", type=int, default=5, help="Command cycles to subtract from the probe")
    parser.add_argument("--source", default="measured with scripts/gen_dma_probe_programs.py")
    parser.add_argument("--min-size", type=int, default=128,
                        help="Drop background points with smaller transfers: a few single-beat commands "
                             "drain faster than any tile stream, which would set a peak no workload reaches")
    args = parser.parse_args()

    ghz, beat = args.frequency_ghz, args.beat_bytes
    curves: dict[str, list[list[float]]] = defaultdict(list)
    unloaded: dict[str, list[float]] = {}
    for log in args.logs:
        match = NAME.search(log.name)
        value = DBG1.search(log.read_text())
        if not match or not value:
            print(f"skip {log.name}: not a probe log")
            continue
        rd, bg, size, store_probe = match.groups()
        if int(bg) and int(size) < args.min_size:
            print(f"skip {log.name}: background transfers below --min-size")
            continue
        dbg1 = int(value.group(1))
        probe = (dbg1 & 0xFFFF) - args.overhead
        phase = dbg1 >> 16
        asm = args.asm_dir / f"{match.group(0)}.S"
        bg_bytes = int(BG_BYTES.search(asm.read_text()).group(1)) if asm.exists() else int(bg) * int(size) * 4
        bandwidth_beats = ((bg_bytes + beat) / beat) / max(1, phase)          # delivered, beats per cycle
        point = [bandwidth_beats * beat * ghz * 1e9 / 1e6, probe / ghz]       # MB/s, ns
        print(f"{log.name}: {bandwidth_beats:.4f} beats/cycle, probe {probe} cycles")
        if int(bg):
            curves[rd].append(point)
        else:
            unloaded["0" if store_probe else "100"] = point
    # The unloaded point anchors every curve: the store probe's for 0 % reads, the
    # load probe's for 100 %, and the mix-weighted mean in between.
    for key in list(curves):
        pct = int(key) / 100
        have = {k: v for k, v in unloaded.items()}
        if not have:
            break
        load = have.get("100", have.get("0"))
        store = have.get("0", have.get("100"))
        curves[key].append([pct * load[0] + (1 - pct) * store[0], pct * load[1] + (1 - pct) * store[1]])
    for key in curves:
        curves[key].sort(key=lambda p: -p[0])
    out = {
        "source": args.source,
        "probe": "dma",
        # The probe is one of the DMA engine's own beats, so every loaded latency
        # is its wait behind the engine's backlog, which the backend's pacing
        # already reproduces: apply lead-off latencies only.
        "latencyMode": "unloaded",
        "measuredChannels": 1,
        "accessBytes": beat,
        "frequencyGHz": ghz,
        "curves": dict(curves),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: {', '.join(f'{k}% x{len(v)}' for k, v in curves.items())}")


if __name__ == "__main__":
    main()
