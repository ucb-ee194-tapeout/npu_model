package atlas.mxu

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import atlas.common._
import atlas.ipt.InnerProductTreesTop
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files, Paths}

class NpuModelMatrixTraceHarness extends Module {
  val p = MregParams()
  val io = IO(new Bundle {
    val cmd = Flipped(Valid(new MxuCmd(p.mregIdBits)))
    val readData = Input(Vec(2, UInt(256.W)))
    val reads = Vec(2, Valid(new MregReadReq(p)))
    val writes = Vec(2, Valid(new MregWriteReq(p)))
  })
  val core = Module(new InnerProductTreesTop(InnerProductTreeParams(), p)).io
  core.cmd <> io.cmd
  io.reads(0) := core.mregReadReq0
  io.reads(1) := core.mregReadReq1
  io.writes(0) := core.mregWriteReq0
  io.writes(1) := core.mregWriteReq1
  core.mregReadResp0.valid := RegNext(io.reads(0).valid, false.B)
  core.mregReadResp1.valid := RegNext(io.reads(1).valid, false.B)
  core.mregReadResp0.bits := RegNext(io.readData(0))
  core.mregReadResp1.bits := RegNext(io.readData(1))
}

class NpuModelMatrixTraceTest extends AnyFlatSpec {
  it should "record the actual IPT engine" in {
    val root = Paths.get(sys.env.getOrElse("MILL_WORKSPACE_ROOT", sys.props("user.dir")))
    // (cycle, op, register, accumulator, weight slot, row count M)
    val commands = Seq(
      (1,0,0,0,0,32), (34,2,2,1,0,32), (67,3,6,1,0,32),
      (100,1,8,0,0,32), (133,4,10,0,0,32), (166,5,8,0,0,32),
      (262,4,12,0,0,32), (295,6,8,0,0,32), (391,4,14,0,0,32),
      (424,3,16,0,0,32), (457,0,18,0,1,32), (460,5,20,1,0,32),
      (557,4,22,1,0,32), (600,7,4,1,0,32), (634,9,24,1,0,32),
      (666,8,28,1,0,32), (699,2,2,1,0,32), (730,3,26,1,0,32),
      // Fewer rows: matmuls on one accumulator max(M, 4) cycles apart, on alternating ones M apart
      // (three in flight); pushes and pops of each kind; a weight push takes all rows.
      (790,0,0,0,0,32), (823,5,1,0,0,3), (827,6,2,0,0,3), (830,5,3,1,0,1),
      (831,6,4,0,0,2), (836,4,10,0,0,5), (842,8,12,1,0,2), (847,7,16,0,0,3),
      (856,3,20,0,0,4), (861,2,22,1,0,6), (870,9,24,1,0,6), (880,1,26,0,0,7),
      (890,8,28,0,0,7), (905,0,6,0,1,5))
    val mem = Array.tabulate(64,32) { (bank,row) =>
      if (bank == 2 || bank == 3)
        (0 until 16).foldLeft(BigInt(0))((v,l) => v | (BigInt(0x3d00 + bank*64 + row*4 + l) << (16*l)))
      else (0 until 32).foldLeft(BigInt(0))((v,l) => v | (BigInt(0x20 + ((bank*7+row*3+l) % 48) + (if ((row+l)%3==0) 128 else 0)) << (8*l)))
    }
    // A command's op indexes this list (test_rtl_matrix_traces.py names them the same way).
    val ops = Seq(MxuOp.PushWeight, MxuOp.PushAccFP8, MxuOp.PushAccBF16, MxuOp.PopAccFP8, MxuOp.PopAccBF16,
      MxuOp.Matmul, MxuOp.MatmulAcc, MxuOp.PushAccFP32, MxuOp.PopAccFP32, MxuOp.PopAccBF16Scaled)
    val events = ujson.Arr()
    simulate(new NpuModelMatrixTraceHarness) { dut =>
      dut.io.cmd.valid.poke(false.B)
      dut.io.cmd.bits.op.poke(MxuOp.PushWeight)
      dut.io.cmd.bits.mregId.poke(0.U)
      dut.io.cmd.bits.accSel.poke(false.B)
      dut.io.cmd.bits.weightSlot.poke(false.B)
      dut.io.cmd.bits.scaleE8M0.poke(127.U)
      dut.io.readData.foreach(_.poke(0.U))
      dut.reset.poke(true.B)
      dut.clock.step(2)
      dut.reset.poke(false.B)
      for (cycle <- 1 to 950) {
        val command = commands.find(_._1 == cycle)
        dut.io.cmd.valid.poke(command.nonEmpty.B)
        command.foreach { case (_,op,bank,acc,weight,rows) =>
          dut.io.cmd.bits.op.poke(ops(op))
          dut.io.cmd.bits.mregId.poke(bank.U)
          dut.io.cmd.bits.accSel.poke((acc == 1).B)
          dut.io.cmd.bits.weightSlot.poke((weight == 1).B)
          dut.io.cmd.bits.scaleE8M0.poke((if (ops(op) == MxuOp.PopAccBF16Scaled) 100 else 127).U)
          dut.io.cmd.bits.rows.poke((rows % 32).U)
        }
        val reads = ujson.Arr()
        val writes = ujson.Arr()
        for (port <- 0 until 2) {
          val r = dut.io.reads(port)
          if (r.valid.peek().litToBoolean) {
            val bank = r.bits.mregId.peek().litValue.toInt
            val row = r.bits.row.peek().litValue.toInt
            reads.value += ujson.Arr(bank,row)
            dut.io.readData(port).poke(mem(bank)(row).U)
          } else dut.io.readData(port).poke(0.U)
        }
        for (port <- 0 until 2) {
          val w = dut.io.writes(port)
          if (w.valid.peek().litToBoolean) {
            val bank = w.bits.mregId.peek().litValue.toInt
            val row = w.bits.row.peek().litValue.toInt
            val data = w.bits.data.peek().litValue
            writes.value += ujson.Arr(bank,row,data.toString(16))
            mem(bank)(row) = data
          }
        }
        events.value += ujson.Obj("cycle" -> cycle, "reads" -> reads, "writes" -> writes)
        dut.clock.step()
      }
    }
    val out = ujson.Obj("commands" -> ujson.Arr.from(commands.map { case (c,o,m,a,w,r) => ujson.Arr(c,o,m,a,w,r) }), "events" -> events)
    Files.writeString(root.resolve("npu-model/tests/rtl/ipt_traces.json"), ujson.write(out, indent=2)+"\n")
  }
}
