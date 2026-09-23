package atlas.vector

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import atlas.common._
import atlas.scalar.VpuCmd
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files, Paths}

/** Actual VPU; external MREG responses have the SRAM's one-cycle latency. */
class NpuModelVectorTraceHarness extends Module {
  val p = MregParams()
  val io = IO(new Bundle {
    val cmd = Flipped(Valid(new VpuCmd))
    val readData = Input(Vec(2, UInt(256.W)))
    val reads = Vec(2, Valid(new MregReadReq(p)))
    val writes = Vec(2, Valid(new MregWriteReq(p)))
    val busy = Output(Bool())
    val issueBusy = Output(UInt(31.W))
  })
  val vpu = Module(new VectorEngineTop(VpuParams(), p))
  vpu.io.cmd <> io.cmd
  io.reads(0) := vpu.io.mregReadReq0
  io.reads(1) := vpu.io.mregReadReq1
  io.writes(0) := vpu.io.mregWriteReq0
  io.writes(1) := vpu.io.mregWriteReq1
  vpu.io.mregReadResp0.valid := RegNext(io.reads(0).valid, false.B)
  vpu.io.mregReadResp1.valid := RegNext(io.reads(1).valid, false.B)
  vpu.io.mregReadResp0.bits := RegNext(io.readData(0))
  vpu.io.mregReadResp1.bits := RegNext(io.readData(1))
  io.busy := vpu.io.busy
  io.issueBusy := vpu.io.issueBusy
}

class NpuModelVectorTraceTest extends AnyFlatSpec {
  behavior of "NPU model VPU trace"
  it should "record every physical row access and result" in {
    val root = Paths.get(sys.env.getOrElse("MILL_WORKSPACE_ROOT", sys.props("user.dir")))
    val traces = ujson.Obj()
    val names = Seq("add", "sub", "mul", "rcp", "sqrt", "sin", "cos", "tanh", "log", "exp", "exp2", "square", "cube", "rsum", "csum", "fp8", "fp8pack", "fp8unpack", "relu", "rmax", "rmin", "cmax", "cmin", "pairmax", "pairmin", "mov", "vliOne", "vliCol", "vliRow", "vliAll")
    simulate(new NpuModelVectorTraceHarness) { dut =>
      val scenarios = names.zipWithIndex.filter(_._1 != "fp8") ++
        names.zipWithIndex.filter(x => Set("add", "sub", "mul", "rsum", "csum", "rmax", "rmin", "cmax", "cmin", "pairmax", "pairmin", "fp8pack", "fp8unpack").contains(x._1)).map { case (n,c) => (n + "_special",c) } ++
        Seq("mov_overlap" -> 25, "add_mirrored" -> 0, "mov_inplace" -> 25, "mov_handoff" -> 25)
      for ((name, code) <- scenarios) {
        val mem = Array.tabulate(64, 32) { (bank, row) =>
          (0 until 16).foldLeft(BigInt(0)) { (bits, lane) =>
            val edge = Seq(0,0x8000,1,0x8001,0x007f,0x807f,0x7f80,0xff80,0x7fc1,0xffa1,0x7f7f,0xff7f,0x3f80,0xbf80,0x3c80,0x43ef)
            val value = if(name.endsWith("_special")) edge((bank*3+row+lane)%16) else 0x3e80 + bank * 32 + row * 4 + lane
            bits | (BigInt(value) << (16 * lane))
          }
        }
        dut.io.cmd.valid.poke(false.B)
        dut.io.cmd.bits.op.poke((code + 1).U)
        dut.io.cmd.bits.vs1.poke(0.U)
        dut.io.cmd.bits.vs2.poke((if(name == "add_mirrored") 0 else 2).U)
        dut.io.cmd.bits.vd.poke((if(name == "mov_inplace") 0 else 4).U)
        dut.io.cmd.bits.scaleE8M0.poke(127.U)
        dut.io.cmd.bits.imm.poke(0x3f80.U)
        dut.io.readData.foreach(_.poke(0.U))
        dut.reset.poke(true.B)
        dut.clock.step(2)
        dut.reset.poke(false.B)
        val events = ujson.Arr()
        for (cycle <- 1 to 135) {
          val overlap = name == "mov_overlap" && cycle == 2
          val handoff = name == "mov_handoff" && cycle == 66
          dut.io.cmd.valid.poke((cycle == 1 || overlap || handoff).B)
          if(overlap || handoff) {
            dut.io.cmd.bits.op.poke((if(overlap) 4 else 26).U)
            dut.io.cmd.bits.vs1.poke(8.U)
            dut.io.cmd.bits.vd.poke(10.U)
          }
          val reads = ujson.Arr()
          val writes = ujson.Arr()
          for (port <- 0 until 2) {
            val read = dut.io.reads(port)
            if (read.valid.peek().litToBoolean) {
              val bank = read.bits.mregId.peek().litValue.toInt
              val row = read.bits.row.peek().litValue.toInt
              reads.value += ujson.Arr(bank, row)
              dut.io.readData(port).poke(mem(bank)(row).U)
            } else dut.io.readData(port).poke(0.U)
          }
          for (port <- 0 until 2) {
            val write = dut.io.writes(port)
            if (write.valid.peek().litToBoolean) {
              val bank = write.bits.mregId.peek().litValue.toInt
              val row = write.bits.row.peek().litValue.toInt
              val data = write.bits.data.peek().litValue
              writes.value += ujson.Arr(bank, row, data.toString(16))
              mem(bank)(row) = data
            }
          }
          events.value += ujson.Obj("cycle" -> cycle, "reads" -> reads, "writes" -> writes,
            "busy" -> dut.io.busy.peek().litToBoolean,
            "issue_busy" -> dut.io.issueBusy.peek().litValue.toDouble)
          dut.clock.step()
        }
        traces(name) = events
      }
    }
    Files.writeString(root.resolve("npu-model/tests/rtl/vector_traces.json"), ujson.write(traces, indent = 2) + "\n")
  }
}

/** Exhaustive BF16 truth tables, including all NaNs/subnormals/signed zeros.
  * Run explicitly when the RTL changes; tables are the model's unary datapath.
  */
class NpuModelUnaryTableTest extends AnyFlatSpec {
  it should "enumerate every BF16 input through the real vector lane boxes" in {
    val root = Paths.get(sys.env.getOrElse("MILL_WORKSPACE_ROOT", sys.props("user.dir")))
    val dest = root.resolve("npu-model/npu_model/hardware/data")
    Files.createDirectories(dest)
    val ops = Seq("rcp" -> 4, "sqrt" -> 5, "sin" -> 6, "cos" -> 7,
      "tanh" -> 8, "log" -> 9, "exp" -> 10, "exp2" -> 11,
      "square" -> 12, "cube" -> 13, "relu" -> 19)
    simulate(new NpuModelVectorTraceHarness) { dut =>
      dut.io.cmd.valid.poke(false.B)
      dut.io.cmd.bits.op.poke(0.U)
      dut.io.cmd.bits.vs1.poke(0.U)
      dut.io.cmd.bits.vs2.poke(2.U)
      dut.io.cmd.bits.vd.poke(4.U)
      dut.io.cmd.bits.scaleE8M0.poke(127.U)
      dut.io.cmd.bits.imm.poke(0.U)
      dut.io.readData.foreach(_.poke(0.U))
      dut.reset.poke(true.B)
      dut.clock.step(2)
      dut.reset.poke(false.B)
      for ((name, code) <- ops) {
        val bytes = new Array[Byte](65536 * 2)
        var count = 0
        dut.io.cmd.bits.op.poke(code.U)
        for (chunk <- 0 until 64; cycle <- 0 until 68) {
          dut.io.cmd.valid.poke((cycle == 0).B)
          val read = dut.io.reads(0)
          if (read.valid.peek().litToBoolean) {
            val bank = read.bits.mregId.peek().litValue.toInt
            val row = read.bits.row.peek().litValue.toInt
            val base = chunk * 1024 + bank * 512 + row * 16
            val data = (0 until 16).foldLeft(BigInt(0))((v, lane) => v | (BigInt(base + lane) << (16 * lane)))
            dut.io.readData(0).poke(data.U)
          } else dut.io.readData(0).poke(0.U)
          for (port <- 0 until 2) {
            val w = dut.io.writes(port)
            if (w.valid.peek().litToBoolean) {
              val bank = w.bits.mregId.peek().litValue.toInt
              val row = w.bits.row.peek().litValue.toInt
              val base = chunk * 1024 + (bank - 4) * 512 + row * 16
              val data = w.bits.data.peek().litValue
              for (lane <- 0 until 16) {
                bytes(2 * (base + lane)) = ((data >> (16 * lane)) & 255).toByte
                bytes(2 * (base + lane) + 1) = ((data >> (16 * lane + 8)) & 255).toByte
                count += 1
              }
            }
          }
          dut.clock.step()
        }
        assert(count == 65536, s"$name: $count outputs")
        val stream = new java.util.zip.GZIPOutputStream(Files.newOutputStream(dest.resolve(s"$name.bin.gz")))
        stream.write(bytes)
        stream.close()
        println(s"Recorded all 65536 BF16 inputs for $name")
      }
    }
  }
}
