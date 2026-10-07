package atlas.scalar

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import atlas.common.VmemParams
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files, Paths}

/** Actual ScalarCore with a one-cycle instruction ROM and scalar memory response.
  * External engines are idle except for the explicitly driven DMA busy signal.
  * The scalar memory response models the one-cycle VMEM read after the
  * ScalarCore's registered LSU command, without pulling in TileLink.
  */
class NpuModelScalarTraceHarness extends Module {
  val io = IO(new Bundle {
    val program = Input(Vec(64, UInt(32.W)))
    val start = Input(Bool())
    val dmaBusy = Input(Bool())
    val fetchPc = Output(UInt(32.W))
    val s1Pc = Output(UInt(32.W))
    val fire = Output(Bool())
    val halted = Output(Bool())
    val illegal = Output(Bool())
    val csrValid = Output(Bool())
    val csrData = Output(UInt(32.W))
  })
  val core = Module(new ScalarCore(VmemParams()))
  core.io.imemFetch.rdata := RegNext(io.program(core.io.imemFetch.addr(5, 0)), 0.U)
  core.io.execRun := true.B
  core.io.execRunWrite := io.start
  core.io.dma_busy := VecInit(Seq(io.dmaBusy) ++ Seq.fill(7)(false.B))
  core.io.vpu_status := 0.U.asTypeOf(new VpuStatus)
  core.io.lsu_scalar_busy := RegNext(core.io.scalarMemCmd.valid && !core.io.scalarMemCmd.bits.isStore, false.B)
  core.io.lsu_vload_busy := false.B
  core.io.lsu_vstore_busy := false.B
  core.io.mregReadBusy := 0.U
  core.io.mregWriteBusy := 0.U
  core.io.csrPort.rdata := 0.U
  core.io.scalarMemResp.valid := RegNext(core.io.scalarMemCmd.valid && !core.io.scalarMemCmd.bits.isStore, false.B)
  core.io.scalarMemResp.bits := "h12345678".U
  io.fetchPc := core.io.imemFetch.addr
  io.s1Pc := core.io.csrPort.illegal_pc
  io.fire := core.io.csrPort.inst_retire
  io.halted := core.io.halted
  io.illegal := core.io.csrPort.set_illegal
  io.csrValid := core.io.csrPort.valid
  io.csrData := core.io.csrPort.wdata
}

/** Regenerate npu-model/tests/rtl/scalar_traces.json with the current RTL:
  * ATLAS_SCALAR_TRACE_INPUT=.../scalar_cases.json
  * ATLAS_SCALAR_TRACE_OUTPUT=.../scalar_traces.json
  * ./mill --no-server atlas.test.testOnly atlas.scalar.NpuModelScalarTraceTest
  */
class NpuModelScalarTraceTest extends AnyFlatSpec {
  behavior of "ScalarCore cycle trace"

  it should "record the scalar frontend regression scenarios" in {
    val root = Paths.get(sys.env.getOrElse("MILL_WORKSPACE_ROOT", sys.props("user.dir")))
    val input = sys.env.getOrElse("ATLAS_SCALAR_TRACE_INPUT",
      root.resolve("npu-model/tests/rtl/scalar_cases.json").toString)
    val output = sys.env.getOrElse("ATLAS_SCALAR_TRACE_OUTPUT",
      root.resolve("npu-model/tests/rtl/scalar_traces.json").toString)
    val cases = ujson.read(Files.readString(Paths.get(input)))
    val traces = ujson.Obj()
    simulate(new NpuModelScalarTraceHarness) { dut =>
      for ((name, scenario) <- cases.obj) {
        val words = scenario("words").arr.map(value => BigInt(value.str, 16))
        require(words.length <= 64)
        dut.io.start.poke(false.B)
        dut.io.dmaBusy.poke(false.B)
        for (i <- 0 until 64) dut.io.program(i).poke(words.lift(i).getOrElse(BigInt(0x73)).U)
        dut.reset.poke(true.B)
        dut.clock.step(2)
        dut.reset.poke(false.B)
        dut.io.start.poke(true.B)
        dut.clock.step(1)
        dut.io.start.poke(false.B)
        val cycles = ujson.Arr()
        var finished = false
        var cycle = 1
        val busyCycles = scenario.obj.get("dma_busy_cycles").map(_.arr.map(_.num.toInt).toSet).getOrElse(Set.empty[Int])
        while (!finished && cycle <= 100) {
          dut.io.dmaBusy.poke(busyCycles.contains(cycle).B)
          val row = ujson.Obj(
            "cycle" -> cycle,
            "fetch_pc" -> dut.io.fetchPc.peek().litValue.toInt,
            "s1_pc" -> dut.io.s1Pc.peek().litValue.toInt,
            "s1_fire" -> dut.io.fire.peek().litToBoolean,
            "halted" -> dut.io.halted.peek().litToBoolean,
            "illegal" -> dut.io.illegal.peek().litToBoolean)
          if (dut.io.csrValid.peek().litToBoolean) row("csr_data") = dut.io.csrData.peek().litValue.toDouble
          finished = dut.io.halted.peek().litToBoolean
          dut.clock.step(1)
          row("next_pc") = dut.io.fetchPc.peek().litValue.toInt
          cycles.value += row
          cycle += 1
        }
        assert(finished, s"$name did not halt within 100 cycles")
        traces(name) = cycles
      }
    }
    Files.writeString(Paths.get(output), ujson.write(traces, indent = 2) + "\n")
  }
}
