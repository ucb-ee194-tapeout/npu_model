package atlas.lsu

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import atlas.common._
import atlas.scalar.LsuCmd
import atlas.xlu.{XluEngine,XluCommand}
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files,Paths}

class NpuModelMemoryTraceHarness extends Module {
  val p = MregParams()
  val v = VmemParams()
  val io = IO(new Bundle {
    val cmd = Flipped(Valid(new LsuCmd(v)))
    val scalar = Flipped(Valid(new LsuScalarCmd(v)))
    val xlu = Flipped(Valid(new XluCommand(p.mregIdBits)))
    val reads = Vec(2, Valid(new MregReadReq(p)))
    val writes = Vec(2, Valid(new MregWriteReq(p)))
    val vreads = Vec(2, Valid(new VmemLineReadPort(v)))
    val vwrite = Valid(new VmemLineWritePort(v))
    val swrite = Valid(new MaskedVmemLineWritePort(v))
    val result = Valid(UInt(32.W))
  })
  val lsu = Module(new LSU(v,p))
  val xlu = Module(new XluEngine(p))
  lsu.io.cmd <> io.cmd
  // ScalarCore's registered scalar command and return interfaces.
  lsu.io.scalarCmd.valid := RegNext(io.scalar.valid, false.B)
  lsu.io.scalarCmd.bits := RegNext(io.scalar.bits)
  io.result.valid := RegNext(lsu.io.scalarResp.valid, false.B)
  io.result.bits := RegNext(lsu.io.scalarResp.bits)
  xlu.io.cmd <> io.xlu
  io.reads(0) := lsu.io.mregReadReq
  io.reads(1) := xlu.io.mregReadReq
  io.writes(0) := lsu.io.mregWriteReq
  io.writes(1) := xlu.io.mregWriteReq
  io.vreads(0) := lsu.io.vmemScalarRead
  io.vreads(1) := lsu.io.vmemVecRead
  io.vwrite := lsu.io.vmemVecWrite
  io.swrite := lsu.io.vmemScalarWrite
  def mdata(r: MregReadReq): UInt = Cat((0 until 32).reverse.map(i => (r.mregId*17.U+r.row*3.U+i.U)(7,0)))
  def vdata(r: VmemLineReadPort): UInt = Cat((0 until 32).reverse.map(i => (r.bankIdx*23.U+r.bankAddr*5.U+i.U)(7,0)))
  lsu.io.mregReadResp.valid := RegNext(io.reads(0).valid,false.B)
  lsu.io.mregReadResp.bits := RegNext(mdata(io.reads(0).bits))
  xlu.io.mregReadResp.valid := RegNext(io.reads(1).valid,false.B)
  xlu.io.mregReadResp.bits := RegNext(mdata(io.reads(1).bits))
  lsu.io.vmemScalarReadData.valid := RegNext(io.vreads(0).valid,false.B)
  lsu.io.vmemScalarReadData.bits := RegNext(vdata(io.vreads(0).bits))
  lsu.io.vmemVecReadData.valid := RegNext(io.vreads(1).valid,false.B)
  lsu.io.vmemVecReadData.bits := RegNext(vdata(io.vreads(1).bits))
}

class NpuModelMemoryTraceTest extends AnyFlatSpec {
  it should "trace concurrent LSU scalar/vector paths and XLU transpose" in {
    val events = ujson.Arr()
    simulate(new NpuModelMemoryTraceHarness) { dut =>
      dut.io.cmd.valid.poke(false.B)
      dut.io.cmd.bits.op.poke(0.U)
      dut.io.cmd.bits.mregBank.poke(0.U)
      dut.io.cmd.bits.vmemLineAddr.poke(0.U)
      dut.io.scalar.valid.poke(false.B)
      dut.io.scalar.bits.isStore.poke(false.B)
      dut.io.scalar.bits.byteAddr.poke((2*262144).U)
      dut.io.scalar.bits.wdata.poke("haabbccdd".U)
      dut.io.scalar.bits.wmask.poke(15.U)
      dut.io.xlu.valid.poke(false.B)
      dut.io.xlu.bits.op.poke(0.U)
      dut.io.xlu.bits.srcMregId.poke(4.U)
      dut.io.xlu.bits.dstMregId.poke(6.U)
      dut.reset.poke(true.B)
      dut.clock.step(2)
      dut.reset.poke(false.B)
      for (cycle <- 1 to 70) {
        dut.io.cmd.valid.poke((cycle == 1 || cycle == 2).B)
        dut.io.cmd.bits.op.poke((if(cycle == 1) 1 else 2).U)
        dut.io.cmd.bits.mregBank.poke((if(cycle == 1) 0 else 2).U)
        dut.io.cmd.bits.vmemLineAddr.poke((if(cycle == 1) 0 else 8192).U)
        dut.io.scalar.valid.poke((cycle == 3 || cycle == 10).B)
        dut.io.scalar.bits.isStore.poke((cycle == 10).B)
        dut.io.scalar.bits.byteAddr.poke((2*262144 + (if(cycle==10) 4 else 0)).U)
        dut.io.xlu.valid.poke((cycle == 1).B)
        val reads = ujson.Arr()
        val writes = ujson.Arr()
        val vr = ujson.Arr()
        for (port <- 0 until 2) {
          val r = dut.io.reads(port)
          if(r.valid.peek().litToBoolean) reads.value += ujson.Arr(r.bits.mregId.peek().litValue.toInt,r.bits.row.peek().litValue.toInt)
          val w = dut.io.writes(port)
          if(w.valid.peek().litToBoolean) writes.value += ujson.Arr(w.bits.mregId.peek().litValue.toInt,w.bits.row.peek().litValue.toInt,w.bits.data.peek().litValue.toString(16))
          val r2 = dut.io.vreads(port)
          if(r2.valid.peek().litToBoolean) vr.value += ujson.Arr(r2.bits.bankIdx.peek().litValue.toInt,r2.bits.bankAddr.peek().litValue.toInt)
        }
        val vw = ujson.Arr()
        if(dut.io.vwrite.valid.peek().litToBoolean) {
          val w = dut.io.vwrite.bits
          vw.value += ujson.Arr(w.bankIdx.peek().litValue.toInt,w.bankAddr.peek().litValue.toInt,w.data.peek().litValue.toString(16))
        }
        val sw = ujson.Arr()
        if(dut.io.swrite.valid.peek().litToBoolean) {
          val w = dut.io.swrite.bits
          val mask = (0 until 32).foldLeft(0L)((m,i) => m | (if(w.mask(i).peek().litToBoolean) 1L << i else 0L))
          sw.value += ujson.Arr(w.bankIdx.peek().litValue.toInt,w.bankAddr.peek().litValue.toInt,w.data.peek().litValue.toString(16),mask.toDouble)
        }
        val result = if(dut.io.result.valid.peek().litToBoolean) ujson.Num(dut.io.result.bits.peek().litValue.toDouble) else ujson.Null
        events.value += ujson.Obj("cycle"->cycle,"reads"->reads,"writes"->writes,"vreads"->vr,"vwrites"->vw,"swrites"->sw,"result"->result)
        dut.clock.step()
      }
    }
    val root = Paths.get(sys.env.getOrElse("MILL_WORKSPACE_ROOT",sys.props("user.dir")))
    Files.writeString(root.resolve("npu-model/tests/rtl/memory_traces.json"),ujson.write(events,indent=2)+"\n")
  }
}
