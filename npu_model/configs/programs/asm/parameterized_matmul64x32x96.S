# 64x32x96 matmul: M=64, K=32, N=96 (2 M-tiles, 1 K-tile, 3 N-tiles)
# A-tile caching: load A once per M-tile, reuse across all N-tiles
# Software pipelining: B[n+1] prefetch overlaps with C[m,n] compute
# MXU1 (35c matmul vs 96c on MXU0)
#
# x1  = VMEM_A  (0x2000)        x11 = N_tiles (3)
# x2  = VMEM_B  (0x2400)        x13 = DRAM_B  (0x0800)
# x3  = VMEM_C0 (0x2800)        x14 = DRAM_C base (advances by x9 per M-tile)
# x4  = VMEM_C1 (0x2C00)        x15 = DRAM_A ptr  (advances by x5 per M-tile)
# x5  = 1024  (tile bytes)      x16 = current B-tile DRAM addr for in-flight DMA
# x6  = 2048  (C-tile bytes)    x17 = DRAM_C tile ptr (advances by x6 per N-tile)
# x9  = 6144  (C row stride)    x20 = M-tile counter
# x10 = M_tiles (2)             x21 = N-tile counter
#
# DMA reads registers at execution time (516c after dispatch). x16 is only
# advanced after dma.wait.ch1 confirms the previous transfer is done, so the
# address is stable for the full transfer window.
lui x1, 0x2
lui x2, 0x2
addi x2, x2, 1024
lui x3, 0x3
addi x3, x3, -2048
lui x4, 0x3
addi x4, x4, -1024
addi x5, x0, 1024
lui x6, 0x1
addi x6, x6, -2048
lui x9, 0x2
addi x9, x9, -2048
addi x10, x0, 2
addi x11, x0, 3
lui x13, 0x1
addi x13, x13, -2048
lui x14, 0x1
addi x14, x14, 1024
addi x15, x0, 0
addi x20, x0, 0
dma.config.ch0 x0
dma.wait.ch0
dma.config.ch1 x0
dma.wait.ch1
dma.wait.ch0
dma.wait.ch1
loop_2:
addi x17, x14, 0
addi x16, x13, 0
dma.load.ch0 x1, x15, x5
dma.load.ch1 x2, x16, x5
dma.wait.ch0
vload m0, 0(x1)
delay 34
addi x21, x0, 0
loop_1:
dma.wait.ch1
vload m1, 0(x2)
delay 34
addi x21, x21, 1
bge x21, x11, no_prefetch
addi x0, x0, 0
addi x0, x0, 0
add x16, x16, x5
dma.load.ch1 x2, x16, x5
no_prefetch:
vmatpush.weight.mxu1 w0, m1
delay 32
vmatmul.mxu1 acc0, m0, w0
delay 35
vmatpop.bf16.acc.mxu1 m2, acc0
delay 32
vstore m2, 0(x3)
delay 34
vstore m3, 0(x4)
delay 34
dma.store.ch0 x17, x3, x6
dma.wait.ch0
add x17, x17, x6
blt x21, x11, loop_1
addi x0, x0, 0
addi x0, x0, 0
add x15, x15, x5
add x14, x14, x9
addi x20, x20, 1
blt x20, x10, loop_2
addi x0, x0, 0
addi x0, x0, 0
