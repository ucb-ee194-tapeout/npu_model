lui x1, 0x2
lui x2, 0x3
addi x2, x2, -2048
lui x3, 0x3
lui x4, 0x4
addi x4, x4, -2048
lui x5, 0x4
addi x6, x0, 1024
lui x7, 0x2
addi x7, x7, 1024
lui x8, 0x3
addi x8, x8, -1024
lui x9, 0x3
addi x9, x9, 1024
lui x10, 0x4
addi x10, x10, -1024
lui x11, 0x4
addi x11, x11, 1024
addi x12, x0, 0
lui x13, 0x2
addi x13, x13, -1024
addi x14, x0, 0
addi x15, x0, 2
lui x16, 0x1
addi x16, x16, -2048
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
lui x17, 0x1
dma.load.ch0 x2, x17, x6
dma.load.ch1 x8, x17, x6
dma.wait.ch0
dma.wait.ch1
lui x17, 0x1
addi x17, x17, 1024
dma.load.ch0 x3, x17, x6
dma.load.ch1 x9, x17, x6
vload m4, 0(x2)
delay 34
vload m5, 32(x2)
delay 34
dma.wait.ch0
dma.wait.ch1
lui x17, 0x2
addi x17, x17, -2048
dma.load.ch0 x4, x17, x6
dma.load.ch1 x10, x17, x6
vload m6, 0(x3)
delay 34
vload m7, 32(x3)
delay 34
dma.wait.ch0
dma.wait.ch1
vload m8, 0(x4)
delay 34
vload m9, 32(x4)
delay 34
vli.all m10,  0x3F80
delay 65
vli.all m11,  0x3F80
delay 65
loop_1:
add x17, x12, x6
add x18, x13, x6
dma.load.ch0 x1, x12, x6
dma.load.ch1 x7, x17, x6
dma.wait.ch0
dma.wait.ch1
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
delay 34
vsquare.bf16 m2, m0
delay 66
vmul.bf16 m12, m2, m0
delay 66
vmul.bf16 m12, m12, m4
delay 66
vadd.bf16 m12, m0, m12
delay 66
vmul.bf16 m12, m12, m6
delay 66
vtanh.bf16 m12, m12
delay 66
vadd.bf16 m12, m10, m12
delay 66
vmul.bf16 m12, m12, m8
delay 66
vmul.bf16 m14, m0, m12
delay 66
vstore m14, 0(x5)
delay 34
vstore m15, 32(x5)
delay 34
dma.store.ch0 x13, x5, x6
dma.store.ch1 x18, x11, x6
dma.wait.ch0
dma.wait.ch1
add x12, x12, x16
add x13, x13, x16
addi x14, x14, 1
blt x14, x15, loop_1
addi x0, x0, 0
addi x0, x0, 0