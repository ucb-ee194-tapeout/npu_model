from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


@dataclass
class HardwareParams:
    input_dtype_width: int = 1
    """ Width of input data type in bytes. """

    output_dtype_width: int = 2
    """ Width of output data type in bytes. """

    MT: int = 64
    """ Number of rows in a matrix register file. """

    NT: int = 16
    """ Number of inner-product tree, or width of systolic array, equals NL. """

    KT: int = 32
    """ Number of elements each tree reduces, or height of systolic array, equals DL. """


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

    input_tile_size_bytes = cfg.input_dtype_width * cfg.MT * cfg.KT
    weight_tile_size_bytes = cfg.input_dtype_width * cfg.NT * cfg.KT
    output_tile_size_bytes = cfg.output_dtype_width * cfg.MT * cfg.NT

    # for m in range(0, M, cfg.MT):
    #     for n in range(0, N, cfg.NT):
    #         for k in range(0, K, cfg.KT):
    #             load_input()
    #             load_weight()
    #             matmul()
    #             for mt in range(cfg.MT):
    #                 tick()
    #                 for nt in range(cfg.NT):
    #                     for dt in range(cfg.KT):
    #                         pass
    #         store_output()
    m_loop_iters = np.ceil(M / cfg.MT)
    n_loop_iters = m_loop_iters * np.ceil(N / cfg.NT)
    k_loop_iters = n_loop_iters * np.ceil(K / cfg.KT)
    mt_loop_iters = k_loop_iters * (cfg.MT)

    stats["num_instructions"] = k_loop_iters
    stats["input_load_bytes"] = k_loop_iters * input_tile_size_bytes
    stats["weight_load_bytes"] = k_loop_iters * weight_tile_size_bytes
    stats["cycles"] = mt_loop_iters
    stats["output_store_bytes"] = n_loop_iters * output_tile_size_bytes

    return stats


def simulate_weight_stationary(cfg: HardwareParams, M: int, N: int, K: int) -> dict:
    stats = initialize_stats()

    input_tile_size_bytes = cfg.input_dtype_width * cfg.MT * cfg.KT
    weight_tile_size_bytes = cfg.input_dtype_width * cfg.NT * cfg.KT
    output_tile_size_bytes = cfg.output_dtype_width * cfg.MT * cfg.NT

    # for k in range(0, K, cfg.KT):
    #     for n in range(0, N, cfg.NT):
    #         load_weight()
    #         for m in range(0, M, cfg.MT):
    #             load_input()
    #             matmul()
    #             for mt in range(cfg.MT):
    #                 tick()
    #                 for nt in range(cfg.NT):
    #                     for dt in range(cfg.KT):
    #                         pass
    #             store_output()
    k_loop_iters = np.ceil(K / cfg.KT)
    n_loop_iters = k_loop_iters * np.ceil(N / cfg.NT)
    m_loop_iters = n_loop_iters * np.ceil(M / cfg.MT)
    mt_loop_iters = m_loop_iters * (cfg.MT)

    stats["weight_load_bytes"] = n_loop_iters * weight_tile_size_bytes
    stats["input_load_bytes"] = m_loop_iters * input_tile_size_bytes
    stats["num_instructions"] = m_loop_iters
    stats["cycles"] = mt_loop_iters
    stats["output_store_bytes"] = m_loop_iters * output_tile_size_bytes

    return stats


def simulate_weight_stationary_rf_reuse(
    cfg: HardwareParams,
    M: int,
    N: int,
    K: int,
    reg_set_size: int = 4,
) -> dict:
    stats = initialize_stats()

    input_tile_size_bytes = cfg.input_dtype_width * cfg.MT * cfg.KT
    weight_tile_size_bytes = cfg.input_dtype_width * cfg.NT * cfg.KT
    output_tile_size_bytes = cfg.output_dtype_width * cfg.MT * cfg.NT

    # for k in range(0, K, cfg.KT):
    #     for n in range(0, N, cfg.NT):
    #         load_weight()
    #         for m in range(0, M, cfg.MT * REG_SET_SIZE):
    #             load_input()
    #             for reg_set in range(0, REG_SET_SIZE): we use 4 registers to hold input and result tiles
    #                 matmul()
    #                 for mt in range(cfg.MT):
    #                     tick()
    #                     for nt in range(cfg.NT):
    #                         for dt in range(cfg.KT):
    #                             pass
    #             store_output()
    n_loop_iters = np.ceil(N / cfg.NT)
    k_loop_iters = n_loop_iters * np.ceil(K / cfg.KT)
    m_loop_iters = k_loop_iters * np.ceil(M / cfg.MT)
    mt_loop_iters = m_loop_iters * (cfg.MT)

    stats["weight_load_bytes"] = k_loop_iters * weight_tile_size_bytes
    stats["input_load_bytes"] = m_loop_iters * input_tile_size_bytes // reg_set_size
    stats["num_instructions"] = m_loop_iters
    stats["cycles"] = mt_loop_iters
    stats["output_store_bytes"] = m_loop_iters * output_tile_size_bytes // reg_set_size

    return stats


def _eff_pct(ideal: int, actual: int) -> float | None:
    """Return efficiency as percentage, or None if N/A."""
    if actual == 0:
        return None
    pct = ideal / actual * 100
    return min(pct, 100.0)


def run_case_study(
    name: str, cfg: HardwareParams, M: int, N: int, K: int
) -> dict[str, Any]:
    """Run output-stationary and weight-stationary for one workload and print comparison. Returns data for plotting."""
    total_flops = 2 * M * N * K
    peak_flops_per_cycle = 2 * cfg.NT * cfg.KT
    ideal_cycles = total_flops / peak_flops_per_cycle
    ideal_input_bytes = M * K * cfg.input_dtype_width
    ideal_weight_bytes = N * K * cfg.input_dtype_width
    ideal_output_bytes = M * N * cfg.output_dtype_width

    os_stats = simulate_output_stationary(cfg, M, N, K)
    ws_stats = simulate_weight_stationary(cfg, M, N, K)
    # ws_stats = simulate_weight_stationary_rf_reuse(cfg, M, N, K)

    assert os_stats["cycles"] == ws_stats["cycles"]
    assert os_stats["num_instructions"] == ws_stats["num_instructions"]

    def _eff(ideal: int, actual: int) -> str:
        pct = _eff_pct(ideal, actual)
        return "N/A" if pct is None else f"{pct:.1f}%"

    print(f"\n--- Case study: {name} (M={M}, N={N}, K={K}) ---")
    print(
        f"  cycles: {os_stats['cycles']:,}  instructions: {os_stats['num_instructions']:,}"
    )
    print(f"  Total FLOPs: {total_flops:,}  |  Ideal cycles: {ideal_cycles:,.0f}")

    ws_in = _eff(ideal_input_bytes, ws_stats["input_load_bytes"])
    os_in = _eff(ideal_input_bytes, os_stats["input_load_bytes"])
    ws_wt = _eff(ideal_weight_bytes, ws_stats["weight_load_bytes"])
    os_wt = _eff(ideal_weight_bytes, os_stats["weight_load_bytes"])
    ws_out = _eff(ideal_output_bytes, ws_stats["output_store_bytes"])
    os_out = _eff(ideal_output_bytes, os_stats["output_store_bytes"])
    ws_total = _eff(
        ideal_input_bytes + ideal_weight_bytes + ideal_output_bytes,
        ws_stats["input_load_bytes"]
        + ws_stats["weight_load_bytes"]
        + ws_stats["output_store_bytes"],
    )
    os_total = _eff(
        ideal_input_bytes + ideal_weight_bytes + ideal_output_bytes,
        os_stats["input_load_bytes"]
        + os_stats["weight_load_bytes"]
        + os_stats["output_store_bytes"],
    )
    mem_tbl = [
        [
            "input",
            f"{ws_stats['input_load_bytes']:,}",
            ws_in,
            f"{os_stats['input_load_bytes']:,}",
            os_in,
        ],
        [
            "weight",
            f"{ws_stats['weight_load_bytes']:,}",
            ws_wt,
            f"{os_stats['weight_load_bytes']:,}",
            os_wt,
        ],
        [
            "output",
            f"{ws_stats['output_store_bytes']:,}",
            ws_out,
            f"{os_stats['output_store_bytes']:,}",
            os_out,
        ],
        [
            "total",
            f"{ws_stats['input_load_bytes'] + ws_stats['weight_load_bytes'] + ws_stats['output_store_bytes']:,}",
            ws_total,
            f"{os_stats['input_load_bytes'] + os_stats['weight_load_bytes'] + os_stats['output_store_bytes']:,}",
            os_total,
        ],
    ]
    print("  Memory (bytes | efficiency):")
    print(
        tabulate(
            mem_tbl,
            headers=["", "WS bytes", "WS eff", "OS bytes", "OS eff"],
            tablefmt="simple",
        )
    )

    ws_read_bytes = ws_stats["input_load_bytes"] + ws_stats["weight_load_bytes"]
    ws_write_bytes = ws_stats["output_store_bytes"]
    os_read_bytes = os_stats["input_load_bytes"] + os_stats["weight_load_bytes"]
    os_write_bytes = os_stats["output_store_bytes"]
    ws_total_bytes = ws_read_bytes + ws_write_bytes
    os_total_bytes = os_read_bytes + os_write_bytes
    ideal_total = ideal_input_bytes + ideal_weight_bytes + ideal_output_bytes
    return {
        "name": name,
        "M": M,
        "N": N,
        "K": K,
        "ws_read_bytes": ws_read_bytes,
        "ws_write_bytes": ws_write_bytes,
        "os_read_bytes": os_read_bytes,
        "os_write_bytes": os_write_bytes,
        "ws_total_bytes": ws_total_bytes,
        "os_total_bytes": os_total_bytes,
        "ws_eff_pct": _eff_pct(ideal_total, ws_total_bytes),
        "os_eff_pct": _eff_pct(ideal_total, os_total_bytes),
    }


def plot_case_studies(
    results: list[dict[str, Any]], out_path: str = "reports/dataflow_comparison.png"
) -> None:
    """Plot memory and efficiency comparison across case studies."""
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Memory bytes comparison (read vs write, stacked)
    scale = 1e6  # MB
    ws_read = [r["ws_read_bytes"] / scale for r in results]
    ws_write = [r["ws_write_bytes"] / scale for r in results]
    os_read = [r["os_read_bytes"] / scale for r in results]
    os_write = [r["os_write_bytes"] / scale for r in results]
    ax1.bar(x - width / 2, ws_read, width, label="WS read", color="steelblue")
    ax1.bar(x - width / 2, ws_write, width, bottom=ws_read, label="WS write", color="lightsteelblue")
    ax1.bar(x + width / 2, os_read, width, label="OS read", color="coral")
    ax1.bar(x + width / 2, os_write, width, bottom=os_read, label="OS write", color="lightsalmon")
    ax1.set_ylabel("Memory (MB)")
    ax1.set_title("Memory transfer (WS vs OS)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Efficiency comparison
    ws_eff = [r["ws_eff_pct"] or 0 for r in results]
    os_eff = [r["os_eff_pct"] or 0 for r in results]
    ax2.bar(
        x - width / 2,
        ws_eff,
        width,
        label="WS",
        facecolor=mcolors.to_rgba("steelblue", 0.3),
        edgecolor="steelblue",
        linewidth=1.5,
    )
    ax2.bar(
        x + width / 2,
        os_eff,
        width,
        label="OS",
        facecolor=mcolors.to_rgba("coral", 0.3),
        edgecolor="coral",
        linewidth=1.5,
    )
    ax2.set_ylabel("Efficiency (%)")
    ax2.set_xlabel("Case study")
    ax2.set_title("Memory efficiency (ideal/actual)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    # ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved plot to {out_path}")


def main():
    cfg = HardwareParams(
        input_dtype_width=1,  # FP8
        output_dtype_width=2,  # BF16
        MT=64,  # 64 rows in a matrix register
        NT=16,  # 16 inner-product trees
        KT=32,  # 32 elements per tree
    )

    # FLOPs/cycle = 2 * NL * DL
    compute_throughput_flops_per_cycle = 2 * cfg.NT * cfg.KT

    print(f"PE compute throughput: {compute_throughput_flops_per_cycle} FLOPs/cycle")

    # -------------------------------------------------------------------------
    # Case studies
    # -------------------------------------------------------------------------

    studies = [
        ("GemmaMLP.up_proj", 816, 16384, 2048),
        ("GemmaMLP.down_proj", 816, 2048, 16384),
        ("GemmaAttention.q_proj", 816, 2048, 2048),
        ("SigLIPAttention.self_attn", 256, 1152, 1152),
        ("SigLIPAttention.fc1", 256, 1152, 4304),
        ("SigLIPAttention.fc2", 256, 4304, 1152),
        ("GemmaMLP.up_proj (bs1)", 51, 4096, 1024),
        ("GemmaMLP.down_proj (bs1)", 51, 1024, 4096),
        ("GemmaAttention KxQ", 816, 256, 816),
        ("GemmaAttention PxV", 816, 816, 256),
    ]
    results = []
    for name, M, N, K in studies:
        results.append(run_case_study(name, cfg, M=M, N=N, K=K))

    plot_case_studies(results)


if __name__ == "__main__":
    main()
