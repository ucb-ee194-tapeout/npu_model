lui x1, 0x2
lui x2, 0x3
addi x2, x2, -2048
lui x3, 0x1
addi x3, x3, -2048
addi x4, x0, 0
lui x5, 0x2
addi x6, x0, 0
addi x7, x0, 4
dma.config.ch0 x0
dma.wait.ch0
vli.all m2,  0xBF80
delay 65
vli.all m3,  0xBF80
delay 65
vli.all m4,  0x3F80
delay 65
vli.all m5,  0x3F80
delay 65
loop_1:
dma.load.ch0 x1, x4, x3
dma.wait.ch0
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
delay 34
vmul.bf16 m6, m0, m2
delay 66
vexp.bf16 m8, m6
delay 66
vadd.bf16 m10, m8, m4
delay 66
vrecip.bf16 m12, m10
delay 66
vmul.bf16 m14, m0, m12
delay 66
vstore m14, 0(x2)
delay 34
vstore m15, 32(x2)
delay 34
dma.store.ch0 x5, x2, x3
dma.wait.ch0
add x4, x4, x3
add x5, x5, x3
addi x6, x6, 1
blt x6, x7, loop_1
addi x0, x0, 0
addi x0, x0, 0