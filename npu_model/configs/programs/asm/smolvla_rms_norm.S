lui x1, 0x2
lui x4, 0x5
addi x5, x0, 0
addi x6, x5, 1024
lui x9, 0x1
addi x10, x9, 1024
addi x13, x0, 1024
addi x11, x1, 1024
addi x15, x4, 1024
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x5, x13
dma.load.ch1 x11, x6, x13
vli.all m6, 0x3D00
delay 65
vli.all m7, 0x3D00
delay 65
vli.all m8, 0x3586
delay 65
vli.all m9, 0x3586
delay 65
dma.wait.ch0
dma.wait.ch1
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
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
vstore m6, 0(x4)
delay 34
vstore m7, 32(x4)
delay 34
dma.store.ch0 x9, x4, x13
dma.store.ch1 x10, x15, x13
dma.wait.ch0
dma.wait.ch1
