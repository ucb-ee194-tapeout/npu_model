# 32x64x32 matmul: M=32, K=64, N=32 (1 M-tile, 2 K-tiles, 1 N-tile)
# All 4 tiles (A[k=0], B[k=0], A[k=1], B[k=1]) dispatched back-to-back into
# separate VMEM buffers so K=0 compute overlaps K=1 data loading
# MXU1 (35c matmul vs 96c on MXU0)
#
# x1  = VMEM_A0 (0x2000)   x11 = N_tiles (1)
# x24 = VMEM_A1 (0x2400)   x12 = K_tiles (2)
# x2  = VMEM_B0 (0x2800)   x13 = DRAM_B  (0x0800)
# x25 = VMEM_B1 (0x2C00)   x14 = DRAM_C  (0x1000)
# x3  = VMEM_C0 (0x3000)   x15 = DRAM_A ptr
# x4  = VMEM_C1 (0x3400)   x16 = DRAM_B[n] addr
# x5  = 1024  (tile bytes)  x17 = DRAM_C tile ptr
# x6  = 2048  (C tile bytes)
# x7  = 1024  (B K-stride)
# x8  = 2048  (A M-stride)
# x9  = 2048  (C row stride)
# x10 = M_tiles (1)
# x18 = A[k=0] DRAM addr, x19 = B[k=0] DRAM addr
# x20 = M counter, x21 = N counter
# x26 = A[k=1] DRAM addr, x27 = B[k=1] DRAM addr
#
# ch0 is reused: A[k=0] sets ch0 first, then B[k=1] sets ch0 again after
# ch0 is cleared by the first dma.wait.ch0.
lui x1, 0x2
lui x24, 0x2
addi x24, x24, 1024
lui x2, 0x3
addi x2, x2, -2048
lui x25, 0x3
addi x25, x25, -1024
lui x3, 0x3
lui x4, 0x3
addi x4, x4, 1024
addi x5, x0, 1024
lui x6, 0x1
addi x6, x6, -2048
addi x7, x0, 1024
lui x8, 0x1
addi x8, x8, -2048
lui x9, 0x1
addi x9, x9, -2048
addi x10, x0, 1
addi x11, x0, 1
lui x13, 0x1
addi x13, x13, -2048
lui x14, 0x1
addi x15, x0, 0
addi x20, x0, 0
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
loop_3:
addi x16, x13, 0
addi x17, x14, 0
addi x21, x0, 0
loop_2:
addi x18, x15, 0
add x26, x18, x5
addi x19, x16, 0
add x27, x19, x7
dma.load.ch0 x1, x18, x5
dma.load.ch1 x2, x19, x5
dma.load.ch2 x24, x26, x5
dma.wait.ch0
dma.load.ch0 x25, x27, x5
vload m0, 0(x1)
delay 34
dma.wait.ch1
vload m1, 0(x2)
delay 34
vmatpush.weight.mxu1 w0, m1
delay 32
vmatmul.mxu1 acc0, m0, w0
delay 35
dma.wait.ch2
vload m0, 0(x24)
delay 34
dma.wait.ch0
vload m1, 0(x25)
delay 34
vmatpush.weight.mxu1 w0, m1
delay 32
vmatmul.acc.mxu1 acc0, m0, w0
delay 35
vmatpop.bf16.acc.mxu1 m2, acc0
delay 32
vstore m2, 0(x3)
delay 34
vstore m3, 0(x4)
delay 34
dma.store.ch0 x17, x3, x6
dma.wait.ch0
addi x16, x16, 1024
add x17, x17, x6
addi x21, x21, 1
blt x21, x11, loop_2
addi x0, x0, 0
addi x0, x0, 0
add x15, x15, x8
add x14, x14, x9
addi x20, x20, 1
blt x20, x10, loop_3
addi x0, x0, 0
addi x0, x0, 0
