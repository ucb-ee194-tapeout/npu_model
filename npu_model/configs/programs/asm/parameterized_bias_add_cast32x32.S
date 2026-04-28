lui x1, 0x2
lui x2, 0x3
addi x2, x2, -2048
lui x3, 0x3
lui x4, 0x1
addi x4, x4, -2048
addi x5, x0, 1024
seli e0, 1
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
addi x6, x0, 0
lui x7, 0x1
addi x7, x7, -2048
dma.load.ch0 x1, x6, x4
dma.load.ch1 x2, x7, x4
dma.wait.ch0
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
delay 34
dma.wait.ch1
vload m2, 0(x2)
delay 34
vload m3, 32(x2)
delay 34
vadd.bf16 m4, m0, m2
delay 66
vpack.bf16.fp8 m6, m4, e0
delay 66
vstore m6, 0(x3)
delay 34
lui x8, 0x1
dma.store.ch0 x8, x3, x5
dma.wait.ch0