seli e0, 1
lui x1, 0x8
lui x2, 0x9
lui x3, 0xa
lui x4, 0xb
lui x5, 0xc
lui x7, 0xe
lui x9, 0x1
lui x10, 0x2
lui x11, 0x3
lui x12, 0x4
lui x14, 0x6
lui x15, 0x1
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x0, x15
dma.load.ch1 x2, x9, x15
dma.load.ch2 x4, x11, x15
vli.all m6,  0x3E00
delay 65
vli.all m7,  0x3E00
delay 65
vli.all m8,  0xC2C8
delay 65
vli.all m9,  0xC2C8
delay 65
vli.all m10, 0
delay 65
vli.all m11, 0
delay 65
vli.all m12, 0
delay 65
vli.all m13, 0
delay 65
vli.all m14, 0
delay 65
vli.all m15, 0
delay 65
dma.wait.ch0
dma.wait.ch1
dma.wait.ch2
dma.load.ch0 x3, x10, x15
dma.load.ch1 x5, x12, x15
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
delay 34
vload m2, 64(x1)
delay 34
vload m3, 96(x1)
delay 34
vpack.bf16.fp8 m4, m0, e0
delay 66
vpack.bf16.fp8 m5, m2, e0
delay 66
vload m16, 0(x2)
delay 34
vload m17, 32(x2)
delay 34
vload m18, 64(x2)
delay 34
vload m19, 96(x2)
delay 34
vpack.bf16.fp8 m20, m16, e0
delay 66
vpack.bf16.fp8 m22, m18, e0
delay 66
vload m24, 0(x4)
delay 34
vload m25, 32(x4)
delay 34
vload m26, 64(x4)
delay 34
vload m27, 96(x4)
delay 34
vpack.bf16.fp8 m28, m24, e0
delay 66
vpack.bf16.fp8 m30, m26, e0
delay 66
vmatpush.weight.mxu0 w0, m20
delay 32
vmatmul.mxu0 acc0, m4, w0
delay 96
vmatpush.weight.mxu0 w0, m22
delay 32
vmatmul.acc.mxu0 acc0, m5, w0
delay 96
vmatpop.bf16.acc.mxu0 m32, acc0
delay 32
vmul.bf16 m34, m32, m6
delay 66
vredmax.row.bf16 m36, m34
delay 34
vmaximum.bf16 m38, m8, m36
delay 66
vsub.bf16 m36, m8, m38
delay 66
vexp.bf16 m36, m36
delay 66
vmul.bf16 m12, m12, m36
delay 66
vmul.bf16 m14, m14, m36
delay 66
vsub.bf16 m40, m34, m38
delay 66
vexp.bf16 m40, m40
delay 66
vpack.bf16.fp8 m42, m40, e0
delay 66
vmatpush.weight.mxu1 w0, m28
delay 32
vmatmul.mxu1 acc0, m42, w0
vmul.bf16 m48, m36, m10
delay 35
vmatpop.bf16.acc.mxu1 m44, acc0
delay 32
vmatpush.weight.mxu1 w0, m30
delay 32
vmatmul.mxu1 acc0, m42, w0
vredsum.row.bf16 m50, m40
delay 35
vmatpop.bf16.acc.mxu1 m46, acc0
delay 32
vadd.bf16 m12, m12, m44
delay 66
vadd.bf16 m14, m14, m46
delay 66
vadd.bf16 m10, m48, m50
delay 66
vmov m8, m38
delay 66
vmov m9, m39
delay 66
dma.wait.ch0
vload m16, 0(x3)
delay 34
vload m17, 32(x3)
delay 34
vload m18, 64(x3)
delay 34
vload m19, 96(x3)
delay 34
vpack.bf16.fp8 m20, m16, e0
delay 66
vpack.bf16.fp8 m22, m18, e0
delay 66
vmatpush.weight.mxu0 w0, m20
delay 32
vmatmul.mxu0 acc0, m4, w0
delay 96
vmatpush.weight.mxu0 w0, m22
delay 32
vmatmul.acc.mxu0 acc0, m5, w0
delay 96
vmatpop.bf16.acc.mxu0 m32, acc0
delay 32
vmul.bf16 m34, m32, m6
delay 66
vredmax.row.bf16 m36, m34
delay 34
vmaximum.bf16 m38, m8, m36
delay 66
vsub.bf16 m36, m8, m38
delay 66
vexp.bf16 m36, m36
delay 66
vmul.bf16 m12, m12, m36
delay 66
vmul.bf16 m14, m14, m36
delay 66
vsub.bf16 m40, m34, m38
delay 66
vexp.bf16 m40, m40
delay 66
vpack.bf16.fp8 m42, m40, e0
delay 66
dma.wait.ch1
vload m24, 0(x5)
delay 34
vload m25, 32(x5)
delay 34
vload m26, 64(x5)
delay 34
vload m27, 96(x5)
delay 34
vpack.bf16.fp8 m28, m24, e0
delay 66
vpack.bf16.fp8 m30, m26, e0
delay 66
vmatpush.weight.mxu1 w0, m28
delay 32
vmatmul.mxu1 acc0, m42, w0
vmul.bf16 m48, m36, m10
delay 35
vmatpop.bf16.acc.mxu1 m44, acc0
delay 32
vmatpush.weight.mxu1 w0, m30
delay 32
vmatmul.mxu1 acc0, m42, w0
vredsum.row.bf16 m50, m40
delay 35
vmatpop.bf16.acc.mxu1 m46, acc0
delay 32
vadd.bf16 m12, m12, m44
delay 66
vadd.bf16 m14, m14, m46
delay 66
vadd.bf16 m10, m48, m50
delay 66
vrecip.bf16 m48, m10
delay 66
vmul.bf16 m12, m12, m48
delay 66
vmul.bf16 m14, m14, m48
delay 66
vstore m12, 0(x7)
delay 34
vstore m13, 32(x7)
delay 34
vstore m14, 64(x7)
delay 34
vstore m15, 96(x7)
delay 34
dma.store.ch0 x14, x7, x15
dma.wait.ch0
