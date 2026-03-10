#include "npu/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npu/Dialect/Kernel/IR/KernelDialect.h"
#include "npu/Dialect/Schedule/IR/ScheduleDialect.h"

namespace npu {

namespace {

struct LowerKernelMatmulToSchedulePattern
    : public mlir::OpRewritePattern<mlir::npu_kernel::MatmulOp> {
  using mlir::OpRewritePattern<mlir::npu_kernel::MatmulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::npu_kernel::MatmulOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::npu_schedule::MatmulTileOp>(
        op, op->getResultTypes(), op.getLhs(), op.getRhs());
    return mlir::success();
  }
};

struct LowerKernelUKernelToSchedulePattern
    : public mlir::OpRewritePattern<mlir::npu_kernel::UKernelGenericOp> {
  using mlir::OpRewritePattern<mlir::npu_kernel::UKernelGenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::npu_kernel::UKernelGenericOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::npu_schedule::UKernelLaunchOp>(
        op, op.getResult().getType(), op.getSymbol(), op.getInputs());
    return mlir::success();
  }
};

struct ConvertNPUKernelToSchedulePass
    : public mlir::PassWrapper<ConvertNPUKernelToSchedulePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNPUKernelToSchedulePass)

  llvm::StringRef getArgument() const final { return "convert-npu-kernel-to-schedule"; }
  llvm::StringRef getDescription() const final {
    return "Lower npu_kernel.matmul/ukernel_generic to npu_schedule ops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::npu_kernel::NPUKernelDialect>();
    registry.insert<mlir::npu_schedule::NPUScheduleDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerKernelMatmulToSchedulePattern>(&getContext());
    patterns.add<LowerKernelUKernelToSchedulePattern>(&getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertNPUKernelToSchedulePass() {
  return std::make_unique<ConvertNPUKernelToSchedulePass>();
}

void registerConvertNPUKernelToSchedulePass() {
  mlir::PassRegistration<ConvertNPUKernelToSchedulePass>();
}

} // namespace npu
