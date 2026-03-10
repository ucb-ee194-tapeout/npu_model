module {
  func.func @matmul_generic_f8f8f32(%arg0: tensor<64x32xf8E4M3FN>, %arg1: tensor<32x16xf8E4M3FN>) -> tensor<64x16xf32> {
    npu_isa.dma_load rd = 2, base = 0, size = 2048, flag = 0
    npu_isa.dma_wait flag = 0
    npu_isa.dma_load_mxu0 rd = 1, base = 2048, size = 512, flag = 0
    npu_isa.dma_wait flag = 0
    %0 = npu_isa.matmul_mxu0 %arg0, %arg1 regs = (0, 2, 1) : tensor<64x32xf8E4M3FN>, tensor<32x16xf8E4M3FN> -> tensor<64x16xf32>
    return %0 : tensor<64x16xf32>
  }
}

