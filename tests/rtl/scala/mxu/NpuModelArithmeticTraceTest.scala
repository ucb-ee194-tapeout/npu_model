package atlas.mxu

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import atlas.common.InnerProductTreeParams
import atlas.ipt.AnchorAccumulationTree
import sp26FPUnits.E4M3FMA
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files,Paths}

class NpuModelArithmeticHarness extends Module {
  val io = IO(new Bundle {
    val a = Input(Vec(32,UInt(8.W)))
    val w = Input(Vec(32,UInt(8.W)))
    val partial = Input(UInt(16.W))
    val scale = Input(UInt(8.W))
    val fma = Output(UInt(16.W))
    val ipt = Output(UInt(16.W))
    val quant = Output(UInt(8.W))
    val dequant = Output(UInt(16.W))
  })
  val fma = Module(new E4M3FMA)
  fma.io.a := io.a(0)
  fma.io.b := io.w(0)
  fma.io.addend16 := io.partial
  io.fma := fma.io.out16
  val ipt = Module(new AnchorAccumulationTree(InnerProductTreeParams()))
  ipt.io.act := io.a
  ipt.io.weightBuf0 := io.w
  ipt.io.weightBuf1 := io.w
  ipt.io.bufReadSel := false.B
  ipt.io.psum := io.partial
  ipt.io.accumulate := true.B
  io.ipt := ipt.io.out
  io.quant := FPUtils.bf16ToE4m3(io.partial,io.scale)
  io.dequant := FPUtils.e4m3ToBf16(io.a(0))
}

class NpuModelArithmeticTraceTest extends AnyFlatSpec {
  it should "record random encodings and arithmetic boundary cases" in {
    val random = new scala.util.Random(90222L)
    val edges = Seq(0,0x8000,1,0x8001,0x007f,0x807f,0x7f80,0xff80,0x7fc1,0xffa1,0x7f7f,0xff7f,0x3f80,0xbf80,0x3c80,0x43ef)
    val cases = ujson.Arr()
    simulate(new NpuModelArithmeticHarness) { dut =>
      dut.io.a.foreach(_.poke(0.U))
      dut.io.w.foreach(_.poke(0.U))
      dut.io.partial.poke(0.U)
      dut.io.scale.poke(127.U)
      dut.reset.poke(true.B)
      dut.clock.step(2)
      dut.reset.poke(false.B)
      for (index <- 0 until 512) {
        val a = (0 until 32).map(lane => if(index < 256 && lane == 0) index else random.nextInt(256))
        val w = (0 until 32).map(_ => random.nextInt(256))
        val partial = if(index < 256) edges(index%16) else random.nextInt(65536)
        val scale = index%256
        a.zipWithIndex.foreach { case(v,l) => dut.io.a(l).poke(v.U) }
        w.zipWithIndex.foreach { case(v,l) => dut.io.w(l).poke(v.U) }
        dut.io.partial.poke(partial.U)
        dut.io.scale.poke(scale.U)
        dut.clock.step(2)
        cases.value += ujson.Obj("a"->ujson.Arr.from(a),"w"->ujson.Arr.from(w),"partial"->partial,"scale"->scale,
          "fma"->dut.io.fma.peek().litValue.toInt,"ipt"->dut.io.ipt.peek().litValue.toInt,
          "quant"->dut.io.quant.peek().litValue.toInt,"dequant"->dut.io.dequant.peek().litValue.toInt)
      }
    }
    val root = Paths.get(sys.env.getOrElse("MILL_WORKSPACE_ROOT",sys.props("user.dir")))
    Files.writeString(root.resolve("npu-model/tests/rtl/arithmetic.json"),ujson.write(cases)+"\n")
  }
}
