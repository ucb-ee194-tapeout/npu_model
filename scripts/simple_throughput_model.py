from dataclasses import dataclass

import numpy as np
from tabulate import tabulate


@dataclass
class HardwareParams:
    input_dtype_width: int = 1
    """ Width of input data type in bytes. """

    output_dtype_width: int = 2
    """ Width of output data type in bytes. """

    NL: int = 16
    """ Number of inner-product tree, or width of systolic array. """

    DL: int = 32
    """ Number of elements each tree reduces, or height of systolic array. """

    ML: int = 64
    """ Number of rows in a matrix register file. """


def initialize_stats() -> dict:
    return {
        "cycles": 0,
        "num_instructions": 0,
        "input_load_bytes": 0,
        "weight_load_bytes": 0,
        "output_store_bytes": 0,
    }


def simulate_output_stationary(cfg: HardwareParams, M: int, N: int, K: int) -> dict:
    stats = initialize_stats()

    input_tile_size_bytes = cfg.input_dtype_width * cfg.ML * cfg.DL
    weight_tile_size_bytes = cfg.input_dtype_width * cfg.NL * cfg.DL
    output_tile_size_bytes = cfg.output_dtype_width * cfg.ML * cfg.NL

    # for m in range(0, M, cfg.ML):
    #     for n in range(0, N, cfg.NL):
    #         for k in range(0, K, cfg.DL):
    #             for mt in range(cfg.ML):
    #                 for nt in range(cfg.NL):
    #                     for dt in range(cfg.DL):
    #                         pass
    m_loop_iters = np.ceil(M / cfg.ML)
    n_loop_iters = m_loop_iters * np.ceil(N / cfg.NL)
    k_loop_iters = n_loop_iters * np.ceil(K / cfg.DL)
    mt_loop_iters = k_loop_iters * (cfg.ML)

    stats["num_instructions"] = k_loop_iters
    stats["input_load_bytes"] = k_loop_iters * input_tile_size_bytes
    stats["weight_load_bytes"] = k_loop_iters * weight_tile_size_bytes
    stats["cycles"] = mt_loop_iters
    stats["output_store_bytes"] = n_loop_iters * output_tile_size_bytes

    return stats


def simulate_weight_stationary(cfg: HardwareParams, M: int, N: int, K: int) -> dict:
    stats = initialize_stats()

    input_tile_size_bytes = cfg.input_dtype_width * cfg.ML * cfg.DL
    weight_tile_size_bytes = cfg.input_dtype_width * cfg.NL * cfg.DL
    output_tile_size_bytes = cfg.output_dtype_width * cfg.ML * cfg.NL

    # for n in range(0, N, cfg.NL):
    #     for k in range(0, K, cfg.DL):
    #         for m in range(0, M, cfg.ML):
    #             for mt in range(cfg.ML):
    #                 for nt in range(cfg.NL):
    #                     for dt in range(cfg.DL):
    #                         pass
    n_loop_iters = np.ceil(N / cfg.NL)
    k_loop_iters = n_loop_iters * np.ceil(K / cfg.DL)
    m_loop_iters = k_loop_iters * np.ceil(M / cfg.ML)
    mt_loop_iters = m_loop_iters * (cfg.ML)

    stats["weight_load_bytes"] = k_loop_iters * weight_tile_size_bytes
    stats["input_load_bytes"] = m_loop_iters * input_tile_size_bytes
    stats["num_instructions"] = m_loop_iters
    stats["cycles"] = mt_loop_iters
    stats["output_store_bytes"] = m_loop_iters * output_tile_size_bytes

    return stats


def run_case_study(name: str, cfg: HardwareParams, M: int, N: int, K: int) -> None:
    """Run output-stationary and weight-stationary for one workload and print comparison."""
    total_flops = 2 * M * N * K
    peak_flops_per_cycle = 2 * cfg.NL * cfg.DL
    ideal_cycles = total_flops / peak_flops_per_cycle
    ideal_input_bytes = M * K * cfg.input_dtype_width
    ideal_weight_bytes = N * K * cfg.input_dtype_width
    ideal_output_bytes = M * N * cfg.output_dtype_width

    os_stats = simulate_output_stationary(cfg, M, N, K)
    ws_stats = simulate_weight_stationary(cfg, M, N, K)

    assert os_stats["cycles"] == ws_stats["cycles"]
    assert os_stats["num_instructions"] == ws_stats["num_instructions"]

    def _eff(ideal: int, actual: int) -> str:
        if actual == 0:
            return "N/A"
        pct = ideal / actual * 100
        return f"{min(pct, 100):.1f}%" if pct > 100 else f"{pct:.1f}%"

    print(f"\n--- Case study: {name} (M={M}, N={N}, K={K}) ---")
    print(
        f"  cycles: {os_stats['cycles']:,}  instructions: {os_stats['num_instructions']:,}"
    )
    print(f"  Total FLOPs: {total_flops:,}  |  Ideal cycles: {ideal_cycles:,.0f}")

    os_in, ws_in = _eff(ideal_input_bytes, os_stats["input_load_bytes"]), _eff(
        ideal_input_bytes, ws_stats["input_load_bytes"]
    )
    os_wt, ws_wt = _eff(ideal_weight_bytes, os_stats["weight_load_bytes"]), _eff(
        ideal_weight_bytes, ws_stats["weight_load_bytes"]
    )
    os_out, ws_out = _eff(ideal_output_bytes, os_stats["output_store_bytes"]), _eff(
        ideal_output_bytes, ws_stats["output_store_bytes"]
    )
    os_total, ws_total = _eff(
        ideal_input_bytes + ideal_weight_bytes + ideal_output_bytes,
        os_stats["input_load_bytes"]
        + os_stats["weight_load_bytes"]
        + os_stats["output_store_bytes"],
    ), _eff(
        ideal_input_bytes + ideal_weight_bytes + ideal_output_bytes,
        ws_stats["input_load_bytes"]
        + ws_stats["weight_load_bytes"]
        + ws_stats["output_store_bytes"],
    )
    mem_tbl = [
        [
            "input",
            f"{os_stats['input_load_bytes']:,}",
            os_in,
            f"{ws_stats['input_load_bytes']:,}",
            ws_in,
        ],
        [
            "weight",
            f"{os_stats['weight_load_bytes']:,}",
            os_wt,
            f"{ws_stats['weight_load_bytes']:,}",
            ws_wt,
        ],
        [
            "output",
            f"{os_stats['output_store_bytes']:,}",
            os_out,
            f"{ws_stats['output_store_bytes']:,}",
            ws_out,
        ],
        [
            "total",
            f"{os_stats['input_load_bytes'] + os_stats['weight_load_bytes'] + os_stats['output_store_bytes']:,}",
            os_total,
            f"{ws_stats['input_load_bytes'] + ws_stats['weight_load_bytes'] + ws_stats['output_store_bytes']:,}",
            ws_total,
        ],
    ]
    print("  Memory (bytes | efficiency):")
    print(
        tabulate(
            mem_tbl,
            headers=["", "OS bytes", "OS eff", "WS bytes", "WS eff"],
            tablefmt="simple",
        )
    )


def main():
    cfg = HardwareParams(
        input_dtype_width=1,  # FP8
        output_dtype_width=2,  # BF16
        NL=16,  # 16 inner-product trees
        DL=32,  # 32 elements per tree
        ML=64,  # 64 rows in a matrix register
    )

    # FLOPs/cycle = 2 * NL * DL
    compute_throughput_flops_per_cycle = 2 * cfg.NL * cfg.DL

    print(f"PE compute throughput: {compute_throughput_flops_per_cycle} FLOPs/cycle")

    # -------------------------------------------------------------------------
    # Case studies
    # -------------------------------------------------------------------------

    run_case_study("GemmaMLP.up_proj", cfg, M=816, N=16384, K=2048)
    run_case_study("GemmaMLP.down_proj", cfg, M=816, N=2048, K=16384)
    run_case_study("GemmaAttention.q_proj", cfg, M=816, N=2048, K=2048)

    run_case_study("SigLIPAttention.self_attn", cfg, M=256, N=1152, K=1152)
    run_case_study("SigLIPAttention.fc1", cfg, M=256, N=1152, K=4304)
    run_case_study("SigLIPAttention.fc2", cfg, M=256, N=4304, K=1152)

    run_case_study("GemmaMLP.up_proj", cfg, M=51, N=4096, K=1024)
    run_case_study("GemmaMLP.down_proj", cfg, M=51, N=1024, K=4096)

    run_case_study("GemmaAttention KxQ", cfg, M=816, N=256, K=816)
    run_case_study("GemmaAttention PxV", cfg, M=816, N=816, K=256)


if __name__ == "__main__":
    main()
