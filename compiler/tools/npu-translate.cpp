#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "npu/Translation/TranslateToTextISA.h"

#include <cstdlib>

int main(int argc, char **argv) {
  npu::registerToTextISATranslation();

  return mlir::failed(
             mlir::mlirTranslateMain(argc, argv, "NPU translation driver\n"))
             ? EXIT_FAILURE
             : EXIT_SUCCESS;
}
