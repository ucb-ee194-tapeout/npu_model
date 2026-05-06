# 64x64x64 matmul: M=64, K=64, N=64 (2 M-tiles, 2 K-tiles, 2 N-tiles)
# A-tile caching: load both K-tiles of A once per M-tile, reuse across N-tiles
# B K-tiles dispatched back-to-back into separate VMEM buffers to eliminate
# the inter-K-tile dispatch gap (~35c per N-tile)
# MXU1 (35c matmul vs 96c on MXU0)
#
# x1  = VMEM_A0 (0x2000)   x11 = N_tiles (2)
# x2  = VMEM_A1 (0x2400)   x13 = DRAM_B  (0x1000)
# x3  = VMEM_B0 (0x2800)   x14 = DRAM_C  (0x2000)
# x23 = VMEM_B1 (0x2C00)   x15 = DRAM_A ptr (advances by x8 per M-tile)
# x4  = VMEM_C0 (0x3000)   x16 = DRAM_B[k=0,n] addr
# x5  = VMEM_C1 (0x3400)   x17 = DRAM_C tile ptr
# x6  = 2048  (B K-stride = C tile bytes)
# x7  = 1024  (FP8 tile bytes)
# x8  = 2048  (A M-stride: K_tiles * tile_bytes)
# x9  = 4096  (C row stride: N_tiles * C tile bytes)
# x10 = M_tiles (2)
# x18 = DRAM_A k=0 ptr, x19 = DRAM_A k=1 ptr
# x20 = M-tile counter, x21 = N-tile counter
# x22 = DRAM_B[k=1,n] addr (x16 + x6, computed before B[k=0] dispatch)
#
# DMA reads register values at execution time (~516c after dispatch).
# x16 and x22 are only modified after dma.wait on the previous transfer.
lui x1, 0x2
lui x2, 0x2
addi x2, x2, 1024
lui x3, 0x3
addi x3, x3, -2048
lui x23, 0x3
addi x23, x23, -1024
lui x4, 0x3
lui x5, 0x3
addi x5, x5, 1024
addi x7, x0, 1024
lui x6, 0x1
addi x6, x6, -2048
lui x8, 0x1
addi x8, x8, -2048
lui x9, 0x1
addi x10, x0, 2
addi x11, x0, 2
lui x13, 0x1
lui x14, 0x2
addi x15, x0, 0
addi x20, x0, 0
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
loop_3:
addi x18, x15, 0
add x19, x15, x7
addi x16, x13, 0
addi x17, x14, 0
dma.load.ch0 x1, x18, x7
dma.load.ch1 x2, x19, x7
dma.wait.ch0
dma.wait.ch1
vload m0, 0(x1)
delay 34
vload m1, 0(x2)
delay 34
addi x21, x0, 0
loop_2:
add x22, x16, x6
dma.load.ch0 x3, x16, x7
dma.load.ch1 x23, x22, x7
dma.wait.ch0
vload m2, 0(x3)
delay 34
vmatpush.weight.mxu1 w0, m2
delay 32
vmatmul.mxu1 acc0, m0, w0
delay 35
dma.wait.ch1
vload m2, 0(x23)
delay 34
vmatpush.weight.mxu1 w0, m2
delay 32
vmatmul.acc.mxu1 acc0, m1, w0
delay 35
vmatpop.bf16.acc.mxu1 m4, acc0
delay 32
vstore m4, 0(x4)
delay 34
vstore m5, 0(x5)
delay 34
dma.store.ch0 x17, x4, x6
dma.wait.ch0
add x17, x17, x6
add x16, x16, x7
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
