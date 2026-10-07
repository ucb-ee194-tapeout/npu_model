package atlas.mxu

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import atlas.common._
import atlas.sa.SystolicArrayTop
import atlas.ipt.InnerProductTreesTop
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files, Paths}

class NpuModelMatrixTraceHarness(sa: Boolean) extends Module {
  val p = MregParams()
  val io = IO(new Bundle {
    val cmd = Flipped(Valid(new MxuCmd(p.mregIdBits)))
    val readData = Input(Vec(2, UInt(256.W)))
    val reads = Vec(2, Valid(new MregReadReq(p)))
    val writes = Vec(2, Valid(new MregWriteReq(p)))
  })
  val core = if (sa) Module(new SystolicArrayTop(SystolicArrayParams(), p)).io
             else Module(new InnerProductTreesTop(InnerProductTreeParams(), p)).io
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
  for (sa <- Seq(true, false)) {
    it should s"record the actual ${if(sa) "SA" else "IPT"} engine" in {
      val root = Paths.get(sys.env.getOrElse("MILL_WORKSPACE_ROOT", sys.props("user.dir")))
      val commands = Seq(
        (1,0,0,0,0), (34,2,2,1,0), (67,3,6,1,0),
        (100,1,8,0,0), (133,4,10,0,0), (166,5,8,0,0),
        (262,4,12,0,0), (295,6,8,0,0), (391,4,14,0,0),
        (424,3,16,0,0), (457,0,18,0,1), (460,5,20,1,0),
        (557,4,22,1,0))
      val mem = Array.tabulate(64,32) { (bank,row) =>
        if (bank == 2 || bank == 3)
          (0 until 16).foldLeft(BigInt(0))((v,l) => v | (BigInt(0x3d00 + bank*64 + row*4 + l) << (16*l)))
        else (0 until 32).foldLeft(BigInt(0))((v,l) => v | (BigInt(0x20 + ((bank*7+row*3+l) % 48) + (if ((row+l)%3==0) 128 else 0)) << (8*l)))
      }
      val events = ujson.Arr()
      simulate(new NpuModelMatrixTraceHarness(sa)) { dut =>
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
        for (cycle <- 1 to 592) {
          val command = commands.find(_._1 == cycle)
          dut.io.cmd.valid.poke(command.nonEmpty.B)
          command.foreach { case (_,op,bank,acc,weight) =>
            dut.io.cmd.bits.op.poke(MxuOp.all.find(_.litValue == op).get)
            dut.io.cmd.bits.mregId.poke(bank.U)
            dut.io.cmd.bits.accSel.poke((acc == 1).B)
            dut.io.cmd.bits.weightSlot.poke((weight == 1).B)
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
      val out = ujson.Obj("commands" -> ujson.Arr.from(commands.map { case (c,o,m,a,w) => ujson.Arr(c,o,m,a,w) }), "events" -> events)
      val name = if(sa) "sa" else "ipt"
      Files.writeString(root.resolve(s"npu-model/tests/rtl/${name}_traces.json"), ujson.write(out, indent=2)+"\n")
    }
  }
}
