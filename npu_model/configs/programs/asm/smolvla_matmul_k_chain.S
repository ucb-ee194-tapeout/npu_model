addi x1, x0, 0
addi x2, x1, 1024
addi x3, x2, 1024
addi x4, x3, 1024
addi x5, x4, 1024
addi x6, x0, 0
addi x7, x6, 256
addi x8, x7, 256
addi x9, x8, 256
addi x10, x9, 256
addi x11, x10, 256
addi x12, x0, 1024
addi x13, x12, 1024
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config.ch0 x30
dma.wait.ch0
dma.load.ch0 x6, x1, x12
dma.load.ch1 x8, x3, x12
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x7, x2, x12
dma.load.ch1 x9, x4, x12
addi x31, x6, 0  # DMA and VLS bases both count words
vload m0, 0(x31)
delay 34
addi x31, x8, 0  # DMA and VLS bases both count words
vload m1, 0(x31)
delay 34
vtrpose.xlu m1, m1   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu1 w0, m1
delay 32
vmatmul.mxu1 acc0, m0, w0
delay 35
dma.wait.ch0
dma.wait.ch1
addi x31, x7, 0  # DMA and VLS bases both count words
vload m2, 0(x31)
delay 34
addi x31, x9, 0  # DMA and VLS bases both count words
vload m3, 0(x31)
delay 34
vtrpose.xlu m3, m3   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu1 w0, m3
delay 32
vmatmul.acc.mxu1 acc0, m2, w0
delay 35
vmatpop.bf16.acc.mxu1 m4, acc0
delay 32
addi x31, x10, 0  # DMA and VLS bases both count words
vstore m4, 0(x31)
delay 34
addi x31, x11, 0  # DMA and VLS bases both count words
vstore m5, 0(x31)
delay 34
dma.store.ch0 x5, x10, x13
dma.wait.ch0
