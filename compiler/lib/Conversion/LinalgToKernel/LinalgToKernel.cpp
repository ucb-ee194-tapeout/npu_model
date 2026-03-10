#include "npu/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npu/Dialect/Kernel/IR/KernelDialect.h"
#include "llvm/Support/raw_ostream.h"

namespace npu {

namespace {

static bool isDimExpr(mlir::AffineExpr expr, unsigned position) {
  auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr);
  return dimExpr && dimExpr.getPosition() == position;
}

static bool isMatmulLikeGeneric(mlir::linalg::GenericOp op) {
  if (!op.hasPureTensorSemantics()) {
    return false;
  }
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1) {
    return false;
  }

  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 3) {
    return false;
  }
  mlir::AffineMap lhsMap = maps[0];
  mlir::AffineMap rhsMap = maps[1];
  mlir::AffineMap outMap = maps[2];
  if (lhsMap.getNumDims() != 3 || rhsMap.getNumDims() != 3 ||
      outMap.getNumDims() != 3) {
    return false;
  }
  if (lhsMap.getNumResults() != 2 || rhsMap.getNumResults() != 2 ||
      outMap.getNumResults() != 2) {
    return false;
  }

  bool lhsMatches = isDimExpr(lhsMap.getResult(0), 0) &&
                    isDimExpr(lhsMap.getResult(1), 2);
  bool rhsMatches = (isDimExpr(rhsMap.getResult(0), 2) &&
                     isDimExpr(rhsMap.getResult(1), 1)) ||
                    (isDimExpr(rhsMap.getResult(0), 1) &&
                     isDimExpr(rhsMap.getResult(1), 2));
  bool outMatches = isDimExpr(outMap.getResult(0), 0) &&
                    isDimExpr(outMap.getResult(1), 1);
  return lhsMatches && rhsMatches && outMatches;
}

static std::string getTypeMnemonic(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  os.flush();

  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    if (llvm::isAlnum(static_cast<unsigned char>(c)) || c == '_') {
      out.push_back(c);
    }
  }
  return out.empty() ? "unknown" : out;
}

static std::string inferMatmulUkernelSymbol(mlir::Value lhs, mlir::Value rhs,
                                            mlir::Type resultType) {
  auto lhsShaped = llvm::dyn_cast<mlir::ShapedType>(lhs.getType());
  auto rhsShaped = llvm::dyn_cast<mlir::ShapedType>(rhs.getType());
  auto outShaped = llvm::dyn_cast<mlir::ShapedType>(resultType);
  if (!lhsShaped || !rhsShaped || !outShaped) {
    return "npu_uk_matmul_generic";
  }

  std::string lhsElem = getTypeMnemonic(lhsShaped.getElementType());
  std::string rhsElem = getTypeMnemonic(rhsShaped.getElementType());
  std::string outElem = getTypeMnemonic(outShaped.getElementType());
  return "npu_uk_matmul_" + lhsElem + "_" + rhsElem + "_" + outElem;
}

struct LowerMatmulToNPUKernelPattern : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
  using mlir::OpRewritePattern<mlir::linalg::MatmulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::MatmulOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    if (op.getNumDpsInputs() != 2 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(op, "expected tensor matmul with 2 inputs and 1 result");
    }

    mlir::Value lhs = op.getDpsInputOperand(0)->get();
    mlir::Value rhs = op.getDpsInputOperand(1)->get();

    rewriter.replaceOpWithNewOp<mlir::npu_kernel::MatmulOp>(
        op, op->getResultTypes(), lhs, rhs);
    return mlir::success();
  }
};

struct LowerMatmulGenericToNPUKernelUKernelPattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using mlir::OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    if (!isMatmulLikeGeneric(op)) {
      return rewriter.notifyMatchFailure(op, "not a matmul-like linalg.generic");
    }

    mlir::Value lhs = op.getDpsInputOperand(0)->get();
    mlir::Value rhs = op.getDpsInputOperand(1)->get();
    auto symbol = rewriter.getStringAttr(
        inferMatmulUkernelSymbol(lhs, rhs, op.getResult(0).getType()));

    rewriter.replaceOpWithNewOp<mlir::npu_kernel::UKernelGenericOp>(
        op, op.getResult(0).getType(), symbol, mlir::ValueRange{lhs, rhs});
    return mlir::success();
  }
};

struct ConvertLinalgToNPUKernelPass
    : public mlir::PassWrapper<ConvertLinalgToNPUKernelPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToNPUKernelPass)

  llvm::StringRef getArgument() const final { return "convert-linalg-to-npu-kernel"; }
  llvm::StringRef getDescription() const final {
    return "Lower linalg.matmul and matmul-like linalg.generic to npu_kernel ops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::npu_kernel::NPUKernelDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerMatmulToNPUKernelPattern>(&getContext());
    patterns.add<LowerMatmulGenericToNPUKernelUKernelPattern>(&getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertLinalgToNPUKernelPass() {
  return std::make_unique<ConvertLinalgToNPUKernelPass>();
}

void registerConvertLinalgToNPUKernelPass() {
  mlir::PassRegistration<ConvertLinalgToNPUKernelPass>();
}

} // namespace npu
