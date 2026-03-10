#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "npu/InitAllDialects.h"
#include "npu/InitAllPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  npu::registerAllDialects(registry);
  npu::registerAllPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NPU optimizer\n", registry));
}
