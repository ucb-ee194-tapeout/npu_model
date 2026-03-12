#include "npu/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npu/Dialect/ISA/IR/ISADialect.h"
#include "npu/Dialect/Schedule/IR/ScheduleDialect.h"
#include "llvm/ADT/StringRef.h"

namespace npu {

namespace {

static mlir::IntegerAttr i64(mlir::PatternRewriter &rewriter, int64_t v) {
  return rewriter.getI64IntegerAttr(v);
}

static mlir::Value emitMatmulSkeleton(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                                      mlir::Type resultType,
                                      mlir::PatternRewriter &rewriter) {
  // Phase-1 deterministic ISA skeleton for a single matmul tile / ukernel launch.
  rewriter.create<mlir::npu_isa::DmaLoadOp>(loc, i64(rewriter, 2), i64(rewriter, 0),
                                             i64(rewriter, 2048), i64(rewriter, 0));
  rewriter.create<mlir::npu_isa::DmaWaitOp>(loc, i64(rewriter, 0));
  rewriter.create<mlir::npu_isa::DmaLoadMxu0Op>(
      loc, i64(rewriter, 1), i64(rewriter, 2048), i64(rewriter, 512), i64(rewriter, 0));
  rewriter.create<mlir::npu_isa::DmaWaitOp>(loc, i64(rewriter, 0));
  auto matmul = rewriter.create<mlir::npu_isa::MatmulMxu0Op>(
      loc, resultType, lhs, rhs, i64(rewriter, 0), i64(rewriter, 2), i64(rewriter, 1));
  return matmul.getResult();
}

static mlir::Value emitSoftmaxVectorChain(mlir::Location loc, mlir::Value input,
                                          mlir::PatternRewriter &rewriter) {
  auto scaled = rewriter.create<mlir::npu_isa::VMulOp>(
      loc, input.getType(), input, input, i64(rewriter, 4), i64(rewriter, 3), i64(rewriter, 2));
  auto exp = rewriter.create<mlir::npu_isa::VExpOp>(
      loc, input.getType(), scaled.getResult(), i64(rewriter, 5), i64(rewriter, 4));
  auto sum = rewriter.create<mlir::npu_isa::VReduceSumOp>(
      loc, input.getType(), exp.getResult(), i64(rewriter, 6), i64(rewriter, 5));
  auto inv = rewriter.create<mlir::npu_isa::VRcpOp>(
      loc, input.getType(), sum.getResult(), i64(rewriter, 7), i64(rewriter, 6));
  auto soft = rewriter.create<mlir::npu_isa::VMulOp>(
      loc, input.getType(), exp.getResult(), inv.getResult(), i64(rewriter, 8), i64(rewriter, 5),
      i64(rewriter, 7));
  return soft.getResult();
}

struct LowerScheduleMatmulToISAPattern
    : public mlir::OpRewritePattern<mlir::npu_schedule::MatmulTileOp> {
  using mlir::OpRewritePattern<mlir::npu_schedule::MatmulTileOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::npu_schedule::MatmulTileOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    mlir::Value result = emitMatmulSkeleton(op.getLoc(), op.getLhs(), op.getRhs(),
                                            op.getResult().getType(), rewriter);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct LowerScheduleSoftmaxToISAPattern
    : public mlir::OpRewritePattern<mlir::npu_schedule::SoftmaxFragmentOp> {
  using mlir::OpRewritePattern<mlir::npu_schedule::SoftmaxFragmentOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::npu_schedule::SoftmaxFragmentOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    mlir::Value result = emitSoftmaxVectorChain(op.getLoc(), op.getInput(), rewriter);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct LowerScheduleUKernelToISAPattern
    : public mlir::OpRewritePattern<mlir::npu_schedule::UKernelLaunchOp> {
  using mlir::OpRewritePattern<mlir::npu_schedule::UKernelLaunchOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::npu_schedule::UKernelLaunchOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    llvm::StringRef symbol = op.getSymbol();
    auto inputs = op.getInputs();

    if (symbol.starts_with("npu_uk_matmul_")) {
      if (inputs.size() < 2) {
        return rewriter.notifyMatchFailure(op, "matmul ukernel expects >=2 tensor inputs");
      }
      mlir::Value result =
          emitMatmulSkeleton(op.getLoc(), inputs[0], inputs[1], op.getResult().getType(), rewriter);
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (symbol.starts_with("npu_uk_gemma_mlp_")) {
      if (inputs.size() < 2) {
        return rewriter.notifyMatchFailure(op, "gemma_mlp ukernel expects >=2 tensor inputs");
      }
      // Closely mirror model_npu/configs/programs/gemma_mlp.py.
      rewriter.create<mlir::npu_isa::DmaLoadMxu0Op>(
          op.getLoc(), i64(rewriter, 0), i64(rewriter, 0x0000), i64(rewriter, 512), i64(rewriter, 0));
      rewriter.create<mlir::npu_isa::DmaLoadMxu0Op>(
          op.getLoc(), i64(rewriter, 1), i64(rewriter, 0x0200), i64(rewriter, 512), i64(rewriter, 1));
      rewriter.create<mlir::npu_isa::DmaWaitOp>(op.getLoc(), i64(rewriter, 0));
      rewriter.create<mlir::npu_isa::DmaWaitOp>(op.getLoc(), i64(rewriter, 1));
      rewriter.create<mlir::npu_isa::DmaLoadOp>(
          op.getLoc(), i64(rewriter, 0), i64(rewriter, 0x2000), i64(rewriter, 2048), i64(rewriter, 2));
      rewriter.create<mlir::npu_isa::DmaWaitOp>(op.getLoc(), i64(rewriter, 2));

      auto gate = rewriter.create<mlir::npu_isa::MatmulMxu0Op>(
          op.getLoc(), op.getResult().getType(), inputs[0], inputs[1], i64(rewriter, 1), i64(rewriter, 0),
          i64(rewriter, 0));
      auto up = rewriter.create<mlir::npu_isa::MatmulMxu0Op>(
          op.getLoc(), op.getResult().getType(), inputs[0], inputs[1], i64(rewriter, 2), i64(rewriter, 0),
          i64(rewriter, 1));
      auto fused = rewriter.create<mlir::npu_isa::VMulOp>(
          op.getLoc(), op.getResult().getType(), gate.getResult(), up.getResult(), i64(rewriter, 6),
          i64(rewriter, 1), i64(rewriter, 2));

      rewriter.create<mlir::npu_isa::DmaStoreOp>(
          op.getLoc(), fused.getResult(), i64(rewriter, 6), i64(rewriter, 0x3000), i64(rewriter, 2048),
          i64(rewriter, 0));
      rewriter.create<mlir::npu_isa::DmaWaitOp>(op.getLoc(), i64(rewriter, 0));

      rewriter.replaceOp(op, fused.getResult());
      return mlir::success();
    }

    if (symbol.starts_with("npu_uk_gemma_attention_")) {
      if (inputs.size() < 2) {
        return rewriter.notifyMatchFailure(op, "gemma_attention ukernel expects >=2 tensor inputs");
      }
      // Closely mirror model_npu/configs/programs/gemma_attention.py.
      rewriter.create<mlir::npu_isa::DmaLoadMxu0Op>(
          op.getLoc(), i64(rewriter, 0), i64(rewriter, 0x2000), i64(rewriter, 256), i64(rewriter, 0));
      rewriter.create<mlir::npu_isa::DmaLoadMxu0Op>(
          op.getLoc(), i64(rewriter, 1), i64(rewriter, 0x3000), i64(rewriter, 256), i64(rewriter, 1));
      rewriter.create<mlir::npu_isa::DmaWaitOp>(op.getLoc(), i64(rewriter, 0));
      rewriter.create<mlir::npu_isa::DmaWaitOp>(op.getLoc(), i64(rewriter, 1));
      rewriter.create<mlir::npu_isa::DmaLoadOp>(
          op.getLoc(), i64(rewriter, 0), i64(rewriter, 0x0000), i64(rewriter, 1024), i64(rewriter, 2));
      rewriter.create<mlir::npu_isa::DmaLoadOp>(
          op.getLoc(), i64(rewriter, 2), i64(rewriter, 0x4000), i64(rewriter, 2048), i64(rewriter, 0));
      rewriter.create<mlir::npu_isa::DmaWaitOp>(op.getLoc(), i64(rewriter, 2));
      rewriter.create<mlir::npu_isa::DmaWaitOp>(op.getLoc(), i64(rewriter, 0));

      auto scores = rewriter.create<mlir::npu_isa::MatmulMxu0Op>(
          op.getLoc(), op.getResult().getType(), inputs[0], inputs[1], i64(rewriter, 3), i64(rewriter, 0),
          i64(rewriter, 0));
      auto softmax = emitSoftmaxVectorChain(op.getLoc(), scores.getResult(), rewriter);
      auto output = rewriter.create<mlir::npu_isa::MatmulMxu0Op>(
          op.getLoc(), op.getResult().getType(), softmax, inputs[1], i64(rewriter, 9), i64(rewriter, 8),
          i64(rewriter, 1));

      rewriter.create<mlir::npu_isa::DmaStoreOp>(
          op.getLoc(), output.getResult(), i64(rewriter, 9), i64(rewriter, 0x5000), i64(rewriter, 2048),
          i64(rewriter, 1));
      rewriter.create<mlir::npu_isa::DmaWaitOp>(op.getLoc(), i64(rewriter, 1));

      rewriter.replaceOp(op, output.getResult());
      return mlir::success();
    }

    if (inputs.size() < 2) {
      return rewriter.notifyMatchFailure(op, "unknown ukernel symbol requires at least 2 tensor inputs");
    }
    // Unknown symbol fallback to generic matmul skeleton.
    mlir::Value result =
        emitMatmulSkeleton(op.getLoc(), inputs[0], inputs[1], op.getResult().getType(), rewriter);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ConvertNPUScheduleToISAPass
    : public mlir::PassWrapper<ConvertNPUScheduleToISAPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNPUScheduleToISAPass)

  llvm::StringRef getArgument() const final { return "convert-npu-schedule-to-isa"; }
  llvm::StringRef getDescription() const final {
    return "Lower npu_schedule matmul/ukernel ops to npu_isa skeleton ops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::npu_schedule::NPUScheduleDialect>();
    registry.insert<mlir::npu_isa::NPUISADialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerScheduleMatmulToISAPattern>(&getContext());
    patterns.add<LowerScheduleSoftmaxToISAPattern>(&getContext());
    patterns.add<LowerScheduleUKernelToISAPattern>(&getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertNPUScheduleToISAPass() {
  return std::make_unique<ConvertNPUScheduleToISAPass>();
}

void registerConvertNPUScheduleToISAPass() {
  mlir::PassRegistration<ConvertNPUScheduleToISAPass>();
}

} // namespace npu
