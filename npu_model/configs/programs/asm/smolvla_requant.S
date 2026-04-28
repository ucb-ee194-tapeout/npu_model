lui x1, 0x2
addi x2, x1, 1024
lui x3, 0x3
addi x4, x0, 0
addi x5, x0, 1024
addi x6, x0, 2047
addi x6, x6, 769
addi x7, x0, 1024
seli e5, 1
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.load.ch0 x1, x4, x7
dma.load.ch1 x2, x5, x7
dma.wait.ch0
dma.wait.ch1
vload m0, 0(x1)
delay 34
vload m1, 0(x2)
delay 34
vpack.bf16.fp8 m2, m0, e5
delay 66
vstore m2, 0(x3)
delay 34
dma.store.ch0 x6, x3, x7
dma.wait.ch0
