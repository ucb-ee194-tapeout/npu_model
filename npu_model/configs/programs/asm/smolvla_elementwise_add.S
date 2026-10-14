lui x1, 0x1
li x2, 0x1400
li x3, 0x1800
addi x4, x0, 0
lui x5, 0x1
lui x6, 0x2
lui x7, 0x1
addi x7, x7, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config.ch0 x30
dma.wait.ch0
dma.config.ch1 x30
dma.wait.ch1
dma.load.ch0 x1, x4, x7
dma.load.ch1 x2, x5, x7
dma.wait.ch0
dma.wait.ch1
addi x31, x1, 0  # DMA and VLS bases both count words
vload m0, 0(x31)
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload m1, 8(x31)
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload m2, 0(x31)
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload m3, 8(x31)
delay 34
vadd.bf16 m4, m0, m2
delay 66
addi x31, x3, 0  # DMA and VLS bases both count words
vstore m4, 0(x31)
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vstore m5, 8(x31)
delay 34
dma.store.ch0 x6, x3, x7
dma.wait.ch0
