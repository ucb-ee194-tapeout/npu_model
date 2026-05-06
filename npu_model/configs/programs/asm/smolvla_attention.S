lui x1, 0x2
addi x2, x1, 1024
addi x3, x2, 1024
lui x6, 0x4
addi x7, x6, 1024
addi x8, x0, 0
addi x9, x8, 1024
addi x10, x9, 1024
lui x12, 0x1
addi x13, x12, 1024
addi x14, x0, 1024
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x8, x14
dma.load.ch1 x2, x9, x14
vli.all m6, 0x3E35
delay 65
vli.all m7, 0x3E35
delay 65
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x3, x10, x14
dma.wait.ch0
vload m0, 0(x1)
delay 34
vload m2, 0(x2)
delay 34
vload m4, 0(x3)
delay 34
vmatpush.weight.mxu0 w0, m2
delay 32
vmatmul.mxu0 acc0, m0, w0
delay 96
vmatpop.bf16.acc.mxu0 m10, acc0
delay 32
vmul.bf16 m12, m10, m6
delay 34
vmatpush.weight.mxu1 w0, m4
delay 32
vredmax.row.bf16 m14, m12
delay 34
vsub.bf16 m16, m12, m14
delay 66
vexp.bf16 m18, m16
delay 66
vredsum.row.bf16 m20, m18
delay 39
vrecip.bf16 m22, m20
delay 66
vmul.bf16 m24, m18, m22
delay 66
seli e0, 1
vpack.bf16.fp8 m26, m24, e0
delay 66
vmatmul.mxu1 acc0, m26, w0
delay 35
vmatpop.bf16.acc.mxu1 m28, acc0
delay 32
vstore m28, 0(x6)
delay 34
vstore m29, 0(x7)
delay 34
dma.store.ch0 x12, x6, x14
dma.store.ch1 x13, x7, x14
dma.wait.ch0
dma.wait.ch1
