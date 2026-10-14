li x1, 0x800
addi x2, x1, 256
li x3, 0xc00
addi x4, x0, 0
addi x5, x0, 1024
addi x6, x0, 2047
addi x6, x6, 769
addi x7, x0, 1024
seli e5, 127
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
addi x31, x2, 0  # DMA and VLS bases both count words
vload m1, 0(x31)
delay 34
vpack.bf16.fp8 m2, m0, e5
delay 66
addi x31, x3, 0  # DMA and VLS bases both count words
vstore m2, 0(x31)
delay 34
dma.store.ch0 x6, x3, x7
dma.wait.ch0
