#include "npu/Conversion/Passes.h"

namespace npu {

void registerConversionPasses() {
  registerConvertLinalgToNPUKernelPass();
  registerConvertNPUKernelToSchedulePass();
  registerConvertNPUScheduleToISAPass();
}

} // namespace npu
