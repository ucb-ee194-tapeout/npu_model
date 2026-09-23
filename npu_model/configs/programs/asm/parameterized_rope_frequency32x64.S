lui x1, 0x2
lui x2, 0x3
addi x2, x2, -2048
lui x3, 0x1
addi x3, x3, -2048
addi x4, x0, 0
lui x5, 0x1
addi x6, x0, 0
addi x7, x0, 2
dma.config.ch0 x0
dma.wait.ch0
loop_1:
dma.load.ch0 x1, x4, x3
dma.wait.ch0
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m0, 0(x31)
delay 34
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m1, 8(x31)
delay 34
vcos.bf16 m2, m0
delay 66
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vstore m2, 0(x31)
delay 34
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vstore m3, 8(x31)
delay 34
dma.store.ch0 x5, x2, x3
dma.wait.ch0
add x4, x4, x3
add x5, x5, x3
addi x6, x6, 1
blt x6, x7, loop_1
addi x0, x0, 0
addi x0, x0, 0
