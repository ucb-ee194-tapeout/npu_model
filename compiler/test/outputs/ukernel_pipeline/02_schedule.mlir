module {
  func.func @matmul_generic_f8f8f32(%arg0: tensor<64x32xf8E4M3FN>, %arg1: tensor<32x16xf8E4M3FN>) -> tensor<64x16xf32> {
    %0 = npu_schedule.ukernel_launch "npu_uk_matmul_f8E4M3FN_f8E4M3FN_f32"(%arg0, %arg1) : tensor<64x32xf8E4M3FN>, tensor<32x16xf8E4M3FN> -> tensor<64x16xf32>
    return %0 : tensor<64x16xf32>
  }
}

