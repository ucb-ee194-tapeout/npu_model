li x1, 0x800
li x2, 0xc00
addi x3, x0, 1024
lui x4, 0x1
addi x4, x4, -1792
addi x5, x0, 0
lui x6, 0x1
addi x6, x6, -2048
addi x7, x0, 0
addi x8, x0, 1
lui x9, 0x1
addi x9, x9, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config.ch0 x30
dma.wait.ch0
dma.config.ch1 x30
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
loop_1:
add x10, x5, x3
dma.load.ch0 x1, x5, x3
dma.load.ch1 x4, x10, x3
dma.wait.ch0
addi x31, x1, 0  # DMA and VLS bases both count words
vload m0, 0(x31)
delay 34
dma.wait.ch1
addi x31, x1, 0  # DMA and VLS bases both count words
vload m1, 8(x31)
delay 34
vredsum.row.bf16 m4, m0
delay 39
addi x31, x2, 0  # DMA and VLS bases both count words
vstore m4, 0(x31)
delay 34
dma.store.ch0 x6, x2, x3
dma.wait.ch0
add x5, x5, x9
add x6, x6, x3
addi x7, x7, 1
blt x7, x8, loop_1
addi x0, x0, 0
addi x0, x0, 0
