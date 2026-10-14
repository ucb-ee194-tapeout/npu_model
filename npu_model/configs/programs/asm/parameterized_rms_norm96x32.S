li x1, 0x800
li x2, 0xc00
lui x3, 0x1
li x4, 0x1400
addi x5, x0, 1024
lui x6, 0x1
addi x6, x6, -1792
lui x7, 0x1
addi x7, x7, -768
lui x8, 0x1
addi x8, x8, 256
lui x9, 0x1
addi x9, x9, 1280
addi x10, x0, 0
lui x11, 0x2
addi x12, x0, 0
addi x13, x0, 3
lui x14, 0x1
addi x14, x14, -2048
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config.ch0 x30
dma.wait.ch0
dma.config.ch1 x30
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
lui x15, 0x2
addi x15, x15, -2048
dma.load.ch0 x2, x15, x5
dma.load.ch1 x7, x15, x5
dma.wait.ch0
dma.wait.ch1
lui x15, 0x2
addi x15, x15, -1024
dma.load.ch0 x3, x15, x5
dma.load.ch1 x8, x15, x5
dma.wait.ch0
dma.wait.ch1
addi x31, x3, 0  # DMA and VLS bases both count words
vload m8, 0(x31)
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vload m9, 8(x31)
delay 34
loop_1:
add x15, x10, x5
add x16, x11, x5
dma.load.ch0 x1, x10, x5
dma.load.ch1 x6, x15, x5
addi x31, x2, 0  # DMA and VLS bases both count words
vload m6, 0(x31)
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload m7, 8(x31)
delay 34
dma.wait.ch0
dma.wait.ch1
addi x31, x1, 0  # DMA and VLS bases both count words
vload m0, 0(x31)
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload m1, 8(x31)
delay 34
vsquare.bf16 m2, m0
delay 66
vredsum.row.bf16 m4, m2
delay 39
vmul.bf16 m10, m4, m6
delay 66
vadd.bf16 m12, m10, m8
delay 66
vsqrt.bf16 m14, m12
delay 66
vrecip.bf16 m4, m14
delay 66
vmul.bf16 m6, m0, m4
delay 66
addi x31, x4, 0  # DMA and VLS bases both count words
vstore m6, 0(x31)
delay 34
addi x31, x4, 0  # DMA and VLS bases both count words
vstore m7, 8(x31)
delay 34
dma.store.ch0 x11, x4, x5
dma.store.ch1 x16, x9, x5
dma.wait.ch0
dma.wait.ch1
add x10, x10, x14
add x11, x11, x14
addi x12, x12, 1
blt x12, x13, loop_1
addi x0, x0, 0
addi x0, x0, 0
