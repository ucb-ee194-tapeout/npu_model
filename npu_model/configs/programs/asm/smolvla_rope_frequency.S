lui x1, 0x1
addi x1, x1, -2048
addi x4, x0, 0
lui x5, 0x1
addi x5, x5, -1280
lui x6, 0x1
addi x6, x6, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config.ch0 x30
dma.wait.ch0
dma.load.ch0 x1, x4, x6
dma.wait.ch0
addi x31, x1, 0  # DMA and VLS bases both count words
vload m0, 0(x31)
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload m1, 8(x31)
delay 34
vcos.bf16 m2, m0
delay 66
addi x31, x1, 0  # DMA and VLS bases both count words
vstore m2, 0(x31)
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vstore m3, 8(x31)
delay 34
dma.store.ch0 x5, x1, x6
dma.wait.ch0
