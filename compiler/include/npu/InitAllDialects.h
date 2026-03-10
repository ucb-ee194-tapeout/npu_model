#ifndef NPU_INIT_ALL_DIALECTS_H_
#define NPU_INIT_ALL_DIALECTS_H_

namespace mlir {
class DialectRegistry;
}

namespace npu {

void registerAllDialects(mlir::DialectRegistry &registry);

} // namespace npu

#endif // NPU_INIT_ALL_DIALECTS_H_
