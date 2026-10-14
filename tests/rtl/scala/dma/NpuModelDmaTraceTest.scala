// Records the actual DmaEngine and Vmem arbiter against a scripted TileLink
// responder and scripted LSU bank traffic. DMA.scala and VMEM.scala depend on
// rocket-chip, so unlike the other npu-model harnesses this one is compiled by
// Chipyard's sbt build (project sp26atlas), not the Mill `atlas` module.
// scripts/regenerate_rtl_fixtures.py installs it temporarily under
// src/test/scala/ and runs it from the Chipyard root.
//
// The responder rule and LSU schedule are repeated in tests/test_rtl_dma_traces.py;
// every per-cycle field here is compared against the Python engine's DmaCycle.
package atlas.dma

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters, TLMessages}
import atlas.common._
import atlas.vmem.Vmem
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files, Paths}
import scala.collection.mutable.ArrayBuffer

class NpuModelDmaTraceHarness extends Module {
  val tp = AtlasParams()
  val v  = tp.vmem
  val bp = TLBundleParameters(
    addressBits = 64, dataBits = tp.dma.beatBytes * 8, sourceBits = tp.dma.tagBits,
    sinkBits = 1, sizeBits = 4, echoFields = Nil, requestFields = Nil, responseFields = Nil,
    hasBCE = false)
  val io = IO(new Bundle {
    val command        = Flipped(Valid(new DmaCommand(v, tp.dma)))
    val channelBusy    = Output(Vec(tp.dma.numChannels, Bool()))
    val tl             = new TLBundle(bp)
    val lsuVecRead     = Flipped(Valid(new VmemLineReadPort(v)))
    val lsuScalarWrite = Flipped(Valid(new MaskedVmemLineWritePort(v)))
    val vmemRead       = Valid(new VmemLineReadPort(v))
    val vmemReadGrant  = Output(Bool())
    val vmemWrite      = Valid(new VmemLineWritePort(v))
    val vmemWriteGrant = Output(Bool())
  })
  val dma  = Module(new DmaEngine(tp, bp))
  val vmem = Module(new Vmem(v, bp))
  dma.io.command := io.command
  io.channelBusy := dma.io.channelBusy
  io.tl <> dma.io.tl
  vmem.io.dmaRead       := dma.io.vmemRead
  dma.io.vmemReadGrant  := vmem.io.dmaReadGrant
  dma.io.vmemReadData   := vmem.io.dmaReadData
  vmem.io.dmaWrite      := dma.io.vmemWrite
  dma.io.vmemWriteGrant := vmem.io.dmaWriteGrant
  io.vmemRead       := dma.io.vmemRead
  io.vmemReadGrant  := vmem.io.dmaReadGrant
  io.vmemWrite      := dma.io.vmemWrite
  io.vmemWriteGrant := vmem.io.dmaWriteGrant
  vmem.io.lsuVecRead     := io.lsuVecRead
  vmem.io.lsuScalarWrite := io.lsuScalarWrite
  vmem.io.lsuScalarRead.valid := false.B
  vmem.io.lsuScalarRead.bits  := DontCare
  vmem.io.lsuVecWrite.valid   := false.B
  vmem.io.lsuVecWrite.bits    := DontCare
  vmem.io.tl.a.valid := false.B
  vmem.io.tl.a.bits  := DontCare
  vmem.io.tl.d.ready := true.B
}

class NpuModelDmaTraceTest extends AnyFlatSpec {
  // (issue cycle, isStore, channel, VMEM line, DRAM address, bytes)
  val commands = Seq(
    (1,  false, 0, 0,        0x1000L, 128),  // four Gets into bank 0
    (2,  true,  1, 8192 + 4, 0x2000L, 96),   // three Puts from bank 1
    (3,  false, 2, 8190,     0x3000L, 256),  // eight Gets crossing bank 0 -> 1
    (4,  true,  3, 0,        0x4000L, 64),   // reads bank 0 under the load writes
    (30, false, 4, 16384,    0x5000L, 64),   // bank 2
    (31, true,  5, 16388,    0x6000L, 128)   // bank 2
  )
  def latency(source: Int): Int = 5 + 2 * (source % 4)     // reorders responses
  def aReady(cycle: Int): Boolean = cycle % 5 != 0           // periodic backpressure
  def lsuVecReadBank(cycle: Int): Option[Int] =
    if (7 <= cycle && cycle <= 12) Some(0) else if (45 <= cycle && cycle <= 47) Some(2) else None
  def lsuScalarWriteBank(cycle: Int): Option[Int] =
    if (cycle == 5 || cycle == 6 || cycle == 14) Some(1) else None
  val cycles = 110

  it should "trace DMA beats, VMEM grants and channel busy against a scripted memory" in {
    val lines = ArrayBuffer[String]()
    simulate(new NpuModelDmaTraceHarness) { dut =>
      val beatBytes = dut.tp.dma.beatBytes
      // Scripted responder: (ready cycle, issue order, source, isStore).
      val pending = ArrayBuffer[(Int, Int, Int, Boolean)]()
      var presented: Option[(Int, Int, Int, Boolean)] = None
      var issued = 0

      dut.io.command.valid.poke(false.B)
      dut.io.tl.a.ready.poke(false.B)
      dut.io.tl.d.valid.poke(false.B)
      dut.io.lsuVecRead.valid.poke(false.B)
      dut.io.lsuScalarWrite.valid.poke(false.B)
      dut.reset.poke(true.B)
      dut.clock.step(2)
      dut.reset.poke(false.B)

      for (cycle <- 1 to cycles) {
        val cmd = commands.find(_._1 == cycle)
        dut.io.command.valid.poke(cmd.isDefined.B)
        cmd.foreach { case (_, isStore, channel, line, dram, bytes) =>
          dut.io.command.bits.opType.poke(if (isStore) DmaDirection.StoreFromVmem else DmaDirection.LoadToVmem)
          dut.io.command.bits.channelId.poke(channel.U)
          dut.io.command.bits.vmemLineAddr.poke(line.U)
          dut.io.command.bits.dramAddress.poke(dram.U)
          dut.io.command.bits.transferSize.poke(bytes.U)
        }
        val vecBank = lsuVecReadBank(cycle)
        dut.io.lsuVecRead.valid.poke(vecBank.isDefined.B)
        dut.io.lsuVecRead.bits.bankIdx.poke(vecBank.getOrElse(0).U)
        dut.io.lsuVecRead.bits.bankAddr.poke(100.U)
        val scalarBank = lsuScalarWriteBank(cycle)
        dut.io.lsuScalarWrite.valid.poke(scalarBank.isDefined.B)
        dut.io.lsuScalarWrite.bits.bankIdx.poke(scalarBank.getOrElse(0).U)
        dut.io.lsuScalarWrite.bits.bankAddr.poke(200.U)
        dut.io.lsuScalarWrite.bits.data.poke(0.U)
        for (i <- 0 until dut.v.lineBytes) dut.io.lsuScalarWrite.bits.mask(i).poke((i < 4).B)

        if (presented.isEmpty && pending.nonEmpty && pending.head._1 <= cycle) presented = Some(pending.head)
        dut.io.tl.d.valid.poke(presented.isDefined.B)
        presented.foreach { case (_, _, source, isStore) =>
          dut.io.tl.d.bits.opcode.poke(if (isStore) TLMessages.AccessAck else TLMessages.AccessAckData)
          dut.io.tl.d.bits.source.poke(source.U)
          dut.io.tl.d.bits.size.poke(log2Ceil(beatBytes).U)
          dut.io.tl.d.bits.param.poke(0.U)
          dut.io.tl.d.bits.sink.poke(0.U)
          dut.io.tl.d.bits.denied.poke(false.B)
          dut.io.tl.d.bits.corrupt.poke(false.B)
          dut.io.tl.d.bits.data.poke((BigInt(source) * 0x0101010101010101L).U)
        }
        dut.io.tl.a.ready.poke(aReady(cycle).B)

        val aFire = dut.io.tl.a.valid.peek().litToBoolean && aReady(cycle)
        val a = if (aFire) {
          val source = dut.io.tl.a.bits.source.peek().litValue.toInt
          val address = dut.io.tl.a.bits.address.peek().litValue
          val isStore = dut.io.tl.a.bits.opcode.peek().litValue == TLMessages.PutFullData.litValue
          val entry = (cycle + latency(source), issued, source, isStore)
          issued += 1
          val index = pending.indexWhere(p => Ordering[(Int, Int)].gt((p._1, p._2), (entry._1, entry._2)))
          if (index < 0) pending += entry else pending.insert(index, entry)
          s"[$source, $address, $isStore]"
        } else "null"
        val dFire = presented.isDefined && dut.io.tl.d.ready.peek().litToBoolean
        val d = if (dFire) {
          val entry = presented.get
          pending -= entry
          presented = None
          entry._3.toString
        } else "null"
        def port(valid: Bool, bankIdx: UInt, bankAddr: UInt, grant: Bool): String =
          if (valid.peek().litToBoolean)
            s"[${bankIdx.peek().litValue}, ${bankAddr.peek().litValue}, ${grant.peek().litToBoolean}]"
          else "null"
        val read  = port(dut.io.vmemRead.valid, dut.io.vmemRead.bits.bankIdx, dut.io.vmemRead.bits.bankAddr, dut.io.vmemReadGrant)
        val write = port(dut.io.vmemWrite.valid, dut.io.vmemWrite.bits.bankIdx, dut.io.vmemWrite.bits.bankAddr, dut.io.vmemWriteGrant)
        val busy = (0 until dut.tp.dma.numChannels).map(ch => dut.io.channelBusy(ch).peek().litToBoolean).mkString("[", ", ", "]")
        lines += s"""  {"cycle": $cycle, "busy": $busy, "a": $a, "d": $d, "vmem_read": $read, "vmem_write": $write}"""
        dut.clock.step()
      }
    }
    val root = Paths.get(sys.env.getOrElse("NPU_MODEL_ROOT",
      Paths.get(sys.props("user.dir"), "generators/sp26-atlas-acc/npu-model").toString))
    Files.writeString(root.resolve("tests/rtl/dma_traces.json"), lines.mkString("[\n", ",\n", "\n]\n"))
  }
}
