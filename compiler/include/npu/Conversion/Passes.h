#ifndef NPU_CONVERSION_PASSES_H_
#define NPU_CONVERSION_PASSES_H_

#include <memory>

namespace mlir {
class Pass;
}

namespace npu {

std::unique_ptr<mlir::Pass> createConvertLinalgToNPUKernelPass();
std::unique_ptr<mlir::Pass> createConvertNPUKernelToSchedulePass();
std::unique_ptr<mlir::Pass> createConvertNPUScheduleToISAPass();

void registerConvertLinalgToNPUKernelPass();
void registerConvertNPUKernelToSchedulePass();
void registerConvertNPUScheduleToISAPass();

void registerConversionPasses();

} // namespace npu

#endif // NPU_CONVERSION_PASSES_H_
