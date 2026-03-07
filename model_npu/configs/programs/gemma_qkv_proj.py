from typing import List, Tuple

import torch

from ...software import Instruction, Program



SEQ_LEN = 64
MODEL_DIM = 32
HEAD_DIM  = 16

INPUT_DATA = torch.ones((SEQ_LEN, MODEL_DIM), dtype=torch.float8_e4m3fn)
Q_WEIGHT_DATA = torch.ones((MODEL_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
K_WEIGHT_DATA = torch.ones((MODEL_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
V_WEIGHT_DATA = torch.ones((MODEL_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)

INPUT_BASE    = 0x0000
Q_WEIGHT_BASE = 0x0800
K_WEIGHT_BASE = 0x0A00
V_WEIGHT_BASE = 0x0C00
Q_OUTPUT_BASE = 0x1000
K_OUTPUT_BASE = 0x1800
V_OUTPUT_BASE = 0x2000


class GemmaQkvProjProgram(Program):
    """
    QKV projection kernel.

    Computes Q = input @ Q_weight, K = input @ K_weight, V = input @ V_weight
    from fp8 input, and stores all three results into DRAM.

    """

    instructions: List[Instruction] = [
        # Load Q_weight and K_weight to WB mxu1 (matmul.mxu0 reads from mxu1)
        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 0,
                "base": Q_WEIGHT_BASE,
                "size": Q_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0
            },
        ),
        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 1,
                "base": K_WEIGHT_BASE,
                "size": K_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 1
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),

        # Load input to MRF 
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 0,
                "base": INPUT_BASE,
                "size": INPUT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),

        # Q projection: input @ Q_weight -> MRF 1 
        Instruction(mnemonic="matmul.mxu0", args={"rd": 1, "rs1": 0, "rs2": 0}),
        # K projection: input @ K_weight -> MRF 2 
        Instruction(mnemonic="matmul.mxu0", args={"rd": 2, "rs1": 0, "rs2": 1}),

        # Load V_weight to WB mxu1 (matmul.mxu0 reads from mxu1)
        Instruction(
            mnemonic="dma.load.mxu0",
            args={
                "rd": 0,
                "base": V_WEIGHT_BASE,
                "size": V_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),

        # V projection: input @ V_weight -> MRF 3
        Instruction(mnemonic="matmul.mxu0", args={"rd": 3, "rs1": 0, "rs2": 0}),

        # Store Q, K, V results
        Instruction(
            mnemonic="dma.store",
            args={"rs1": 1, "base": Q_OUTPUT_BASE,
                  "size": SEQ_LEN * HEAD_DIM * torch.bfloat16.itemsize, "flag": 0},
        ),
        Instruction(
            mnemonic="dma.store",
            args={"rs1": 2, "base": K_OUTPUT_BASE,
                  "size": SEQ_LEN * HEAD_DIM * torch.bfloat16.itemsize, "flag": 1},
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        Instruction(
            mnemonic="dma.store",
            args={"rs1": 3, "base": V_OUTPUT_BASE,
                  "size": SEQ_LEN * HEAD_DIM * torch.bfloat16.itemsize, "flag": 0},
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (INPUT_BASE,    INPUT_DATA),
        (Q_WEIGHT_BASE, Q_WEIGHT_DATA),
        (K_WEIGHT_BASE, K_WEIGHT_DATA),
        (V_WEIGHT_BASE, V_WEIGHT_DATA),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        Q_OUTPUT_BASE,
        (INPUT_DATA.to(torch.float32) @ Q_WEIGHT_DATA.to(torch.float32)).to(torch.bfloat16),
    )
