lui x1, 0x2
addi x2, x1, 1024
addi x3, x2, 1024
addi x4, x3, 1024
lui x5, 0x3
addi x6, x5, 1024
addi x7, x0, 0
addi x8, x0, 1024
addi x9, x8, 1024
addi x10, x9, 1024
lui x11, 0x1
addi x11, x11, -1024
addi x12, x11, 1024
addi x13, x0, 1024
dma.config.ch1 x0
dma.wait.ch1
dma.load.ch0 x1, x7, x13
dma.load.ch1 x2, x8, x13
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x3, x9, x13
dma.load.ch1 x4, x10, x13
dma.wait.ch0
dma.wait.ch1
vload m0, 0(x1)
delay 34
vload m1, 0(x2)
delay 34
vload m2, 0(x3)
delay 34
vload m3, 0(x4)
delay 34
vrecip.bf16 m4, m2
delay 66
vmul.bf16 m6, m0, m4
delay 66
vstore m6, 0(x5)
delay 34
vstore m7, 0(x6)
delay 34
dma.store.ch0 x11, x5, x13
dma.store.ch1 x12, x6, x13
dma.wait.ch0
dma.wait.ch1
