# DMA memory calibration

Everything the measured DMA memory curve (`npu_model/configs/memory_curves/ee290sim_vcs_probe.json`)
was made from, so it can be regenerated or re-measured from this repository alone.

- `programs/perf_dma_*.S`: timing programs in the RTL repo's baremetal assembler
  syntax. Each banks the mcycles delta around a DMA region into CSR_DBG1.
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
