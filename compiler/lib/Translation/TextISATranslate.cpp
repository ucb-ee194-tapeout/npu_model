#include "npu/Translation/TranslateToTextISA.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "npu/Dialect/ISA/IR/ISADialect.h"
#include "npu/InitAllDialects.h"

namespace npu {
namespace {

mlir::LogicalResult translateToTextISA(mlir::Operation *op, llvm::raw_ostream &os) {
  op->walk([&](mlir::Operation *nested) {
    if (auto load = llvm::dyn_cast<mlir::npu_isa::DmaLoadOp>(nested)) {
      os << "dma.load"
         << " rd=" << load.getRd()
         << ", base=" << load.getBase()
         << ", size=" << load.getSize()
         << ", flag=" << load.getFlag() << "\n";
      return;
    }
    if (auto load = llvm::dyn_cast<mlir::npu_isa::DmaLoadMxu0Op>(nested)) {
      os << "dma.load.mxu0"
         << " rd=" << load.getRd()
         << ", base=" << load.getBase()
         << ", size=" << load.getSize()
         << ", flag=" << load.getFlag() << "\n";
      return;
    }
    if (auto load = llvm::dyn_cast<mlir::npu_isa::DmaLoadMxu1Op>(nested)) {
      os << "dma.load.mxu1"
         << " rd=" << load.getRd()
         << ", base=" << load.getBase()
         << ", size=" << load.getSize()
         << ", flag=" << load.getFlag() << "\n";
      return;
    }
    if (auto store = llvm::dyn_cast<mlir::npu_isa::DmaStoreOp>(nested)) {
      os << "dma.store"
         << " rs1=" << store.getRs1()
         << ", base=" << store.getBase()
         << ", size=" << store.getSize()
         << ", flag=" << store.getFlag() << "\n";
      return;
    }
    if (auto wait = llvm::dyn_cast<mlir::npu_isa::DmaWaitOp>(nested)) {
      os << "dma.wait"
         << " flag=" << wait.getFlag() << "\n";
      return;
    }
    if (auto matmul = llvm::dyn_cast<mlir::npu_isa::MatmulMxu0Op>(nested)) {
      os << "matmul.mxu0"
         << " rd=" << matmul.getRd()
         << ", rs1=" << matmul.getRs1()
         << ", rs2=" << matmul.getRs2() << "\n";
      return;
    }
    if (auto vmul = llvm::dyn_cast<mlir::npu_isa::VMulOp>(nested)) {
      os << "vmul"
         << " vrd=" << vmul.getVrd()
         << ", vs1=" << vmul.getVs1()
         << ", vs2=" << vmul.getVs2() << "\n";
      return;
    }
    if (auto vexp = llvm::dyn_cast<mlir::npu_isa::VExpOp>(nested)) {
      os << "vexp"
         << " vrd=" << vexp.getVrd()
         << ", vs1=" << vexp.getVs1() << "\n";
      return;
    }
    if (auto vrsum = llvm::dyn_cast<mlir::npu_isa::VReduceSumOp>(nested)) {
      os << "vreduce.sum"
         << " vrd=" << vrsum.getVrd()
         << ", vs1=" << vrsum.getVs1() << "\n";
      return;
    }
    if (auto vrcp = llvm::dyn_cast<mlir::npu_isa::VRcpOp>(nested)) {
      os << "vrcp"
         << " vrd=" << vrcp.getVrd()
         << ", vs1=" << vrcp.getVs1() << "\n";
      return;
    }
  });

  return mlir::success();
}

} // namespace

void registerToTextISATranslation() {
  mlir::TranslateFromMLIRRegistration reg(
      "mlir-to-npu-text-isa",
      "Translate npu_isa ops to simulator-compatible textual ISA",
      translateToTextISA,
      [](mlir::DialectRegistry &registry) { registerAllDialects(registry); });
}

} // namespace npu
