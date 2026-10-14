li x1, 0x800
lui x2, 0x1
addi x2, x2, -1536
li x3, 0xc00
lui x4, 0x1
addi x4, x4, -2048
addi x5, x0, 1024
seli e0, 127
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config.ch0 x30
dma.wait.ch0
dma.config.ch1 x30
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
addi x6, x0, 0
lui x7, 0x1
addi x7, x7, -2048
dma.load.ch0 x1, x6, x4
dma.load.ch1 x2, x7, x4
dma.wait.ch0
addi x31, x1, 0  # DMA and VLS bases both count words
vload m0, 0(x31)
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload m1, 8(x31)
delay 34
dma.wait.ch1
addi x31, x2, 0  # DMA and VLS bases both count words
vload m2, 0(x31)
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload m3, 8(x31)
delay 34
vadd.bf16 m4, m0, m2
delay 66
vpack.bf16.fp8 m6, m4, e0
delay 66
addi x31, x3, 0  # DMA and VLS bases both count words
vstore m6, 0(x31)
delay 34
lui x8, 0x1
dma.store.ch0 x8, x3, x5
dma.wait.ch0
