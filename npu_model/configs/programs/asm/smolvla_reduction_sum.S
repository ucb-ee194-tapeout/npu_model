lui x1, 0x2
lui x2, 0x3
addi x3, x0, 0
addi x4, x3, 1024
lui x5, 0x1
addi x5, x5, -1280
addi x6, x0, 1024
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.load.ch0 x1, x3, x6
addi x7, x1, 1024
dma.load.ch1 x7, x4, x6
dma.wait.ch0
dma.wait.ch1
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m0, 0(x31)
delay 34
srli x31, x1, 2   # VLS bases count words; DMA bases count bytes
vload m1, 8(x31)
delay 34
vredsum.row.bf16 m4, m0
delay 39
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vstore m4, 0(x31)
delay 34
dma.store.ch0 x5, x2, x6
dma.wait.ch0
