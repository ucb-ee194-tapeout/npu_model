#include "npu/InitAllPasses.h"

#include "npu/Conversion/Passes.h"

namespace npu {

void registerAllPasses() {
  registerConversionPasses();
}

} // namespace npu
