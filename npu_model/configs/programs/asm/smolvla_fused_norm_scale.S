lui x1, 0x2
lui x2, 0x3
lui x3, 0x4
addi x4, x0, 0
lui x5, 0x1
addi x5, x5, -2048
lui x6, 0x1
lui x7, 0x1
addi x7, x7, -2048
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.load.ch0 x1, x4, x7
dma.load.ch1 x2, x5, x7
dma.wait.ch0
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
delay 34
vsqrt.bf16 m4, m0
delay 66
vrecip.bf16 m6, m4
delay 66
dma.wait.ch1
vload m2, 0(x2)
delay 34
vload m3, 32(x2)
delay 34
vmul.bf16 m8, m2, m6
delay 66
vstore m8, 0(x3)
delay 34
vstore m9, 32(x3)
delay 34
dma.store.ch0 x6, x3, x7
dma.wait.ch0
