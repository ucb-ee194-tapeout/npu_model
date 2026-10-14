addi x1, x0, 0
addi x2, x0, 1280
lui x3, 0x1
addi x3, x3, -1280
lui x4, 0x1
addi x4, x4, -2048
addi x5, x4, 256
addi x6, x5, 256
addi x7, x0, 1024
lui x8, 0x1
addi x8, x8, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config.ch0 x30
dma.wait.ch0
dma.load.ch0 x4, x1, x7
dma.load.ch1 x5, x2, x7
dma.wait.ch0
dma.wait.ch1
addi x31, x4, 0  # DMA and VLS bases both count words
vload m0, 0(x31)
delay 34
addi x31, x5, 0  # DMA and VLS bases both count words
vload m1, 0(x31)
delay 34
vtrpose.xlu m1, m1   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu1 w0, m1
delay 32
vmatmul.mxu1 acc0, m0, w0
delay 35
vmatpop.bf16.acc.mxu1 m2, acc0
delay 32
addi x31, x6, 0  # DMA and VLS bases both count words
vstore m2, 0(x31)
delay 34
addi x9, x6, 256
addi x31, x9, 0  # DMA and VLS bases both count words
vstore m3, 0(x31)
delay 34
dma.store.ch0 x3, x6, x8
dma.wait.ch0
