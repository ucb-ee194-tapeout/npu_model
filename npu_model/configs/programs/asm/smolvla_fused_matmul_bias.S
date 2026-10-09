li x1, 0x800
lui x2, 0x1
addi x2, x2, -1792
lui x3, 0x1
addi x3, x3, -1536
li x4, 0xc00
addi x5, x0, 0
addi x6, x0, 1024
lui x7, 0x1
addi x7, x7, -2048
lui x8, 0x1
addi x9, x0, 1024
lui x10, 0x1
addi x10, x10, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config.ch0 x30
dma.wait.ch0
dma.load.ch0 x1, x5, x9
dma.load.ch1 x2, x6, x9
dma.load.ch2 x3, x7, x10
dma.wait.ch0
dma.wait.ch1
addi x31, x1, 0  # DMA and VLS bases both count words
vload m0, 0(x31)
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload m2, 0(x31)
delay 34
vtrpose.xlu m2, m2   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu1 w0, m2
delay 32
vmatmul.mxu1 acc0, m0, w0
dma.wait.ch2
addi x31, x3, 0  # DMA and VLS bases both count words
vload m4, 0(x31)
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vload m5, 8(x31)
delay 34
vmatpop.bf16.acc.mxu1 m6, acc0
delay 32
vadd.bf16 m8, m6, m4
delay 66
addi x31, x4, 0  # DMA and VLS bases both count words
vstore m8, 0(x31)
delay 34
addi x31, x4, 0  # DMA and VLS bases both count words
vstore m9, 8(x31)
delay 34
dma.store.ch0 x8, x4, x10
dma.wait.ch0
