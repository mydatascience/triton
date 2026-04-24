#include <gtest/gtest.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"

namespace mlir::triton::gpu {
namespace {

TEST(CoarseScheduleTest, SplitClusterBeforeDoesNotInsertUnscheduledOps) {
  MLIRContext ctx;
  ctx.loadDialect<scf::SCFDialect, arith::ArithDialect>();

  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();

  ModuleOp mod = ModuleOp::create(loc);
  b.setInsertionPointToStart(mod.getBody());

  auto c0 = arith::ConstantIntOp::create(b, loc, 0, 32);
  auto c1 = arith::ConstantIntOp::create(b, loc, 1, 32);
  auto forOp = scf::ForOp::create(b, loc, c0, c1, c1);

  b.setInsertionPointToStart(forOp.getBody());
  Operation *unscheduledOp =
      arith::ConstantIntOp::create(b, loc, 42, 32).getOperation();
  auto add = arith::AddIOp::create(b, loc, unscheduledOp->getResult(0),
                                   unscheduledOp->getResult(0));
  Operation *scheduledOp = add.getOperation();
  scf::YieldOp::create(b, loc);

  CoarseSchedule schedule(/*numStages=*/1);
  auto cluster = schedule.clusters.newAtBack();
  schedule.insert(scheduledOp, /*stage=*/0, cluster);

  EXPECT_EQ(schedule.count(unscheduledOp), 0);
  schedule.splitClusterBefore(scheduledOp, forOp);
  EXPECT_EQ(schedule.count(unscheduledOp), 0)
      << "splitClusterBefore must not insert default entries for ops not "
         "already in the schedule";
}

} // namespace
} // namespace mlir::triton::gpu
