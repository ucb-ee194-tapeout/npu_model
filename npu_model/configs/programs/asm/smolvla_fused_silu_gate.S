lui x1, 0x2
lui x2, 0x3
addi x3, x0, 0
lui x4, 0x1
addi x4, x4, -2048
lui x5, 0x1
addi x5, x5, -2048
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x3, x5
vli.all m2,  0xBF80
delay 65
vli.all m3,  0xBF80
delay 65
vli.all m4,  0x3F80
delay 65
vli.all m5,  0x3F80
delay 65
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
vmul.bf16 m14, m12, m0
delay 66
vstore m14, 0(x2)
delay 34
vstore m15, 32(x2)
delay 34
dma.store.ch0 x4, x2, x5
dma.wait.ch0
