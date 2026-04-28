lui x1, 0x2
lui x5, 0x6
addi x6, x0, 0
addi x7, x6, 1024
lui x11, 0x1
addi x11, x11, 1024
addi x12, x11, 1024
addi x13, x0, 1024
addi x14, x1, 1024
addi x18, x5, 1024
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x6, x13
dma.load.ch1 x14, x7, x13
vli.all m2, 0x3D37
delay 65
vli.all m3, 0x3D37
delay 65
vli.all m4, 0x3F4C
delay 65
vli.all m5, 0x3F4C
delay 65
vli.all m6, 0x3F00
delay 65
vli.all m7, 0x3F00
delay 65
vli.all m8, 0x3F80
delay 65
vli.all m9, 0x3F80
delay 65
dma.wait.ch0
dma.wait.ch1
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
delay 34
vmul.bf16 m10, m0, m0
delay 66
vmul.bf16 m12, m10, m0
delay 66
vmul.bf16 m14, m12, m2
delay 66
vadd.bf16 m16, m0, m14
delay 66
vmul.bf16 m18, m16, m4
delay 66
vtanh.bf16 m20, m18
delay 66
vadd.bf16 m22, m8, m20
delay 66
vmul.bf16 m24, m0, m6
delay 66
vmul.bf16 m26, m24, m22
delay 66
vstore m26, 0(x5)
delay 34
dma.store.ch0 x11, x5, x13
vstore m27, 0(x18)
delay 34
dma.store.ch1 x12, x18, x13
dma.wait.ch0
dma.wait.ch1
