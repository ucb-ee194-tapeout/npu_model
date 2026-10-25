package atlas.mxu

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import atlas.common.InnerProductTreeParams
import atlas.ipt.AnchorAccumulationTree
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files,Paths}

class NpuModelArithmeticHarness extends Module {
  val io = IO(new Bundle {
    val a = Input(Vec(32,UInt(8.W)))
    val w = Input(Vec(32,UInt(8.W)))
    val partial = Input(UInt(16.W))
    val c = Input(UInt(32.W))
    val scale = Input(UInt(8.W))
    val ipt = Output(UInt(32.W))
    val quant = Output(UInt(8.W))
    val dequant = Output(UInt(16.W))
  })
  val ipt = Module(new AnchorAccumulationTree(InnerProductTreeParams()))
  ipt.io.act := io.a
  ipt.io.weightBuf0 := io.w
  ipt.io.weightBuf1 := io.w
  ipt.io.bufReadSel := false.B
  ipt.io.psum := io.c
  io.ipt := ipt.io.out
  io.quant := FPUtils.bf16ToE4m3(io.partial,io.scale)
  io.dequant := FPUtils.e4m3ToBf16(io.a(0))
}

class NpuModelArithmeticTraceTest extends AnyFlatSpec {
  it should "record random encodings and arithmetic boundary cases" in {
    val random = new scala.util.Random(90222L)
    val edges = Seq(0,0x8000,1,0x8001,0x007f,0x807f,0x7f80,0xff80,0x7fc1,0xffa1,0x7f7f,0xff7f,0x3f80,0xbf80,0x3c80,0x43ef)
    val edges32 = Seq(0L,0x80000000L,1L,0x80000001L,0x007fffffL,0x00800000L,0x7f800000L,0xff800000L,
                      0x7fc00000L,0xffc00001L,0x7f7fffffL,0xff7fffffL,0x3f800000L,0xbf800000L,0x2f800001L,0x4b7fffffL)
    val cases = ujson.Arr()
    simulate(new NpuModelArithmeticHarness) { dut =>
      dut.io.a.foreach(_.poke(0.U))
      dut.io.w.foreach(_.poke(0.U))
      dut.io.partial.poke(0.U)
      dut.io.c.poke(0.U)
      dut.io.scale.poke(127.U)
      dut.reset.poke(true.B)
      dut.clock.step(2)
      dut.reset.poke(false.B)
      for (index <- 0 until 512) {
        val a = (0 until 32).map(lane => if(index < 256 && lane == 0) index else random.nextInt(256))
        val w = (0 until 32).map(_ => random.nextInt(256))
        val partial = if(index < 256) edges(index%16) else random.nextInt(65536)
        // C: edges, then values near the products (exponent field 100..150), then any bits.
        val c = if(index < 256) edges32(index%16)
                else if(index < 384) (random.nextInt(2).toLong << 31) | ((100L + random.nextInt(51)) << 23) | random.nextInt(1 << 23)
                else random.nextLong() & 0xffffffffL
        val scale = index%256
        a.zipWithIndex.foreach { case(v,l) => dut.io.a(l).poke(v.U) }
        w.zipWithIndex.foreach { case(v,l) => dut.io.w(l).poke(v.U) }
        dut.io.partial.poke(partial.U)
        dut.io.c.poke(c.U)
        dut.io.scale.poke(scale.U)
        dut.clock.step(2)
        cases.value += ujson.Obj("a"->ujson.Arr.from(a),"w"->ujson.Arr.from(w),"partial"->partial,"c"->c.toDouble,"scale"->scale,
          "ipt"->dut.io.ipt.peek().litValue.toDouble,
          "quant"->dut.io.quant.peek().litValue.toInt,"dequant"->dut.io.dequant.peek().litValue.toInt)
      }
    }
    val root = Paths.get(sys.env.getOrElse("MILL_WORKSPACE_ROOT",sys.props("user.dir")))
    Files.writeString(root.resolve("npu-model/tests/rtl/arithmetic.json"),ujson.write(cases)+"\n")
  }
}
