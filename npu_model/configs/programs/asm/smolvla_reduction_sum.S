li x1, 0x800
li x2, 0xc00
addi x3, x0, 0
addi x4, x3, 1024
lui x5, 0x1
addi x5, x5, -1280
addi x6, x0, 1024
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config.ch0 x30
dma.wait.ch0
dma.config.ch1 x30
dma.wait.ch1
dma.load.ch0 x1, x3, x6
addi x7, x1, 256
dma.load.ch1 x7, x4, x6
dma.wait.ch0
dma.wait.ch1
addi x31, x1, 0  # DMA and VLS bases both count words
vload m0, 0(x31)
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload m1, 8(x31)
delay 34
vredsum.row.bf16 m4, m0
delay 39
addi x31, x2, 0  # DMA and VLS bases both count words
vstore m4, 0(x31)
delay 34
dma.store.ch0 x5, x2, x6
dma.wait.ch0
