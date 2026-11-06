# DMA memory calibration

Everything the measured DMA memory curve (`npu_model/configs/memory_curves/ee290sim_vcs_probe.json`)
was made from, so it can be regenerated or re-measured from this repository alone.

- `programs/perf_dma_*.S`: timing programs. Each banks the mcycles delta around
  a DMA region into CSR_DBG1.
  `perf_dma_probe_<rd>r_<bg>bg_<size>.S` (38 files, written by
  `scripts/gen_dma_probe_programs.py`) are the DMA-side Mess benchmark: one
  timed 32 B access on channel 0 against background traffic from `bg`
  channels at `rd` percent loads and `size` bytes per transfer. The other 21
  are single and multi-channel transfers that `scripts/calibrate_dma_backend.py`
  scores the backend against.
- `measurements/ee290sim_vcs.json`: `dbg1_cycles` of every program on VCS
  (`chipyard.harness.TestHarness.EE290SimConfig`), refresh off (used) and with
  refresh (reference only). Refresh off means Chipyard's default DRAMSim2 ini
  (`testchipip/src/main/resources/dramsim2_ini`) with one change,
  `REFRESH_PERIOD=100000000` in the DDR3 ini.

Regenerate the curve and the residual table without any RTL run:

    python scripts/fit_dma_probe_curve.py --out npu_model/configs/memory_curves/ee290sim_vcs_probe.json
    python scripts/calibrate_dma_backend.py

Re-measure (VCS or the bringup board): copy `programs/*.S` into the RTL repo's
`baremetal/assembly`, run `baremetal/agentic/run_asm_tests.sh` with
`EXTRA_SIM_FLAGS='+dramsim_ini_dir=<copy of the default ini dir with REFRESH_PERIOD=100000000>'`,
then pass the logs with `--logs` to both scripts.

## Transaction timing (`transactions/`)

`scripts/gen_dma_transaction_programs.py` writes `transactions/txn_*.S`: one
timed transfer per program (CSR_DBG1 covers only the probe, after any warm-up
and `dma.wait`). Families: a cold 32 B load/store; a 32 B probe at A + 2^k
after a 32 B warm-up at A (k = 5..22, loads after loads, stores after stores,
and the two mixed orders at k = 5 and 20); the same-row probe after `DELAY`
0..4000; single cold transfers of 32 B to 4 KiB; and 2 or 4 x 1 KiB on
separate channels or serially on one. `--channel-style bare` writes `0..7`
instead of `ch0..ch7` for the RTL repository's older assembler.

`scripts/dma_transactions.py` runs them on the model and reports per beat
(A-fire to D-fire, launch to first A, last D to retire, the backend's chosen
latency) and per command, merges metasim `summary.csv` or VCS values by
program name, and for single-beat programs derives the measured beat round
trip as the measured CSR_DBG1 minus the model's fixed engine overhead. The
traces come from `DmaExecutionUnit.beat_trace` / `command_trace`.
