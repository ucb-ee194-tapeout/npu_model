lui x1, 0x2
addi x1, x1, 0
addi x4, x0, 0
lui x5, 0x1
addi x5, x5, -1280
lui x6, 0x1
addi x6, x6, -2048
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x4, x6
dma.wait.ch0
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
delay 34
vcos.bf16 m2, m0
delay 66
vstore m2, 0(x1)
delay 34
vstore m3, 32(x1)
delay 34
dma.store.ch0 x5, x1, x6
dma.wait.ch0
