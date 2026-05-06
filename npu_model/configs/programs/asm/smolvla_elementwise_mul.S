lui x1, 0x4
lui x2, 0x5
lui x3, 0x6
addi x4, x0, 0
lui x5, 0x1
lui x6, 0x2
lui x7, 0x1
addi x7, x7, -2048
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x4, x7
dma.load.ch1 x2, x5, x7
dma.wait.ch0
dma.wait.ch1
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
delay 34
vload m2, 0(x2)
delay 34
vload m3, 32(x2)
delay 34
vmul.bf16 m4, m0, m2
delay 66
vstore m4, 0(x3)
delay 34
vstore m5, 32(x3)
delay 34
dma.store.ch0 x6, x3, x7
dma.wait.ch0
