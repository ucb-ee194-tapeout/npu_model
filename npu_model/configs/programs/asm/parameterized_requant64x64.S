lui x1, 0x2
lui x2, 0x2
addi x2, x2, 1024
lui x3, 0x3
addi x4, x0, 1024
addi x5, x0, 0
lui x6, 0x2
addi x7, x0, 0
addi x8, x0, 4
lui x9, 0x1
addi x9, x9, -2048
seli e0, 127
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
loop_1:
add x10, x5, x4
dma.load.ch0 x1, x5, x4
dma.load.ch1 x2, x10, x4
dma.wait.ch0
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m0, 0(x31)
delay 34
dma.wait.ch1
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vload m1, 0(x31)
delay 34
vpack.bf16.fp8 m2, m0, e0
delay 66
srli x31, x3, 2   # VLS bases count words; DMA bases count bytes
vstore m2, 0(x31)
delay 34
dma.store.ch0 x6, x3, x4
dma.wait.ch0
add x5, x5, x9
add x6, x6, x4
addi x7, x7, 1
blt x7, x8, loop_1
addi x0, x0, 0
addi x0, x0, 0
