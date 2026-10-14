"""Clock-edge schedules derived from DMA.scala, Vmem.scala and AtlasCore.scala.

Each test drives the engine directly, so cycle 1 is the launch tick (S1 fire).
The off-chip side is a FixedLatencyBackend whose latency and acceptance are
scripted per test; LSU bank traffic is announced through the conflict checker
exactly as LoadStoreUnit does.
"""
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.isa_definition import (
    DMA_CONFIG_CH0, DMA_LOAD_CH0, DMA_LOAD_CH1, DMA_LOAD_CH2, DMA_STORE_CH1, DMA_STORE_CH3,
)
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.dma import DmaExecutionUnit
from npu_model.hardware.memory_backend import CurveMemoryBackend, FixedLatencyBackend
from npu_model.logging.logger import Logger
from npu_model.software import x
from npu_model.software.instruction import Uop

BEAT = 32
BANK = 256 * 1024
# x1: VMEM word address, x2: DRAM offset, x3: 1 KiB, x4: 64 B, x5: second VMEM tile
# (byte 1024), x6: second DRAM offset, x7: VMEM tile in bank 1 (byte BANK + 2048),
# x8: dma.base value. DMA VMEM operands count 32-bit words, as in AtlasCore.
REGS = {1: 0, 2: 0x100, 3: 1024, 4: 64, 5: 1024 // 4, 6: 0x2000, 7: (BANK + 2048) // 4, 8: 1}


@pytest.fixture
def dma():
    cfg = DefaultHardwareConfig()
    cfg.arch_state_config = replace(cfg.arch_state_config, dram_base=0, dram_size=1 << 20, numerics="rtl")
    state = ArchState(cfg.arch_state_config)
    generator = torch.Generator().manual_seed(0)
    state.dram[:] = torch.randint(0, 256, state.dram.shape, generator=generator, dtype=torch.uint8)  # the whole 1 MiB aperture
    state.vmem[:] = torch.randint(0, 256, state.vmem.shape, generator=generator, dtype=torch.uint8)
    for reg, value in REGS.items():
        state.write_xrf(reg, value)
    unit = DmaExecutionUnit("DMA0", Mock(spec=Logger), state, config=cfg)
    unit.backend = FixedLatencyBackend(latency=4)
    yield unit
    state.close()


def launch(unit, insn):
    """S1 launch: the core sets the channel flag in the same tick."""
    uop = Uop(insn)
    unit.arch_state.set_flag(insn.funct3)
    unit.tick(uop)
    return unit.last_cycle


def tick(unit, lsu_banks=None):
    """One cycle with the LSU touching ``lsu_banks`` (as LoadStoreUnit announces)."""
    if lsu_banks:
        unit.arch_state.conflict_checker.announce_vmem_ports(
            unit.cycle + 1, {bank: "vload" for bank in lsu_banks})
    unit.tick(None)
    return unit.last_cycle


def run(unit, max_cycles=4096):
    events = []
    while unit.has_in_flight and len(events) < max_cycles:
        events.append(tick(unit))
    assert not unit.has_in_flight, "transfer did not complete"
    return events


LOAD = DMA_LOAD_CH0(rd=x(1), rs1=x(2), rs2=x(4))      # 64 B: two beats, DRAM 0x100 -> VMEM 0
STORE = DMA_STORE_CH1(rd=x(6), rs1=x(1), rs2=x(4))    # 64 B: two beats, VMEM 0 -> DRAM 0x2000


def test_load_issues_one_beat_per_cycle_from_the_cycle_after_launch(dma):
    first = launch(dma, LOAD)                       # T: command valid, slot active from T+1
    assert first.a is None and first.busy == [False] * 8
    second = tick(dma)                              # T+1: first Get
    assert second.a == [0, 0x100, False] and second.busy[0]
    third = tick(dma)                               # T+2: second Get
    assert third.a == [1, 0x120, False]
    assert dma.in_flight and dma._slots[0].dispatched


def test_load_writes_each_beat_as_its_response_arrives(dma):
    state = dma.arch_state
    expected = state.dram[0x100:0x140].clone()
    launch(dma, LOAD)
    tick(dma)                                       # T+1: Get beat 0
    tick(dma)                                       # T+2: Get beat 1
    tick(dma); tick(dma)                            # T+3, T+4
    assert not torch.equal(state.vmem[0:32], expected[:32])
    event = tick(dma)                               # T+5: beat 0 data (latency 4)
    assert event.d == 0 and event.vmem_write == [0, 0, True]
    assert torch.equal(state.vmem[0:32], expected[:32])
    assert not torch.equal(state.vmem[32:64], expected[32:])
    event = tick(dma)                               # T+6: beat 1 data
    assert event.d == 1
    assert torch.equal(state.vmem[0:64], expected)


def test_channel_busy_clears_the_cycle_after_the_last_beat_retires(dma):
    state = dma.arch_state
    launch(dma, LOAD)
    for _ in range(5):
        tick(dma)                                   # through T+5: beat 0 acknowledged
    assert state.check_flag(0)
    event = tick(dma)                               # T+6: last D fire, slot dispatched -> retire
    assert event.d == 1 and event.busy[0]
    assert not state.check_flag(0)                  # S1 sees busy low from T+7
    assert not dma.has_in_flight
    assert dma.complete_count == 1
    assert tick(dma).busy == [False] * 8


def test_source_bytes_are_read_when_each_beat_is_requested(dma):
    state = dma.arch_state
    launch(dma, LOAD)
    tick(dma)                                       # T+1: beat 0 requested
    state.dram[0x100:0x140] = 7                     # Beat 0 already sampled; beat 1 not yet.
    original = state.dram[0x100:0x120].clone()
    run(dma)
    assert not torch.equal(state.vmem[0:32], torch.full((32,), 7, dtype=torch.uint8))
    assert torch.equal(state.vmem[32:64], torch.full((32,), 7, dtype=torch.uint8))
    assert torch.equal(state.dram[0x100:0x120], original)


def test_store_reads_vmem_ahead_and_requests_two_cycles_after_the_grant(dma):
    state = dma.arch_state
    expected = state.vmem[0:64].clone()
    launch(dma, STORE)
    event = tick(dma)                               # T+1: VMEM read beat 0 granted
    assert event.vmem_read == [0, 0, True] and event.a is None
    event = tick(dma)                               # T+2: read beat 1; data 0 in SyncReadMem
    assert event.vmem_read == [0, 1, True] and event.a is None
    event = tick(dma)                               # T+3: beat 0 leaves storeDataQueue -> Put
    assert event.a == [0, 0x2000, True] and event.vmem_read is None
    assert torch.equal(state.dram[0x2000:0x2020], expected[:32])
    event = tick(dma)                               # T+4: Put beat 1
    assert event.a == [1, 0x2020, True]
    assert torch.equal(state.dram[0x2000:0x2040], expected)
    for _ in range(3):
        tick(dma)                                   # T+5..T+7: acks at latency 4
    assert dma.last_cycle.d == 0
    assert state.check_flag(1)
    tick(dma)                                       # T+8: last ack retires the slot
    assert dma.last_cycle.d == 1 and not state.check_flag(1)


def test_store_data_is_what_vmem_held_when_the_line_was_read(dma):
    state = dma.arch_state
    launch(dma, STORE)
    tick(dma)                                       # T+1: line 0 read
    state.vmem[0:64] = 9                            # Line 1 still unread.
    run(dma)
    assert not torch.equal(state.dram[0x2000:0x2020], torch.full((32,), 9, dtype=torch.uint8))
    assert torch.equal(state.dram[0x2020:0x2040], torch.full((32,), 9, dtype=torch.uint8))


def test_lsu_access_denies_the_dma_write_and_holds_channel_d(dma):
    state = dma.arch_state
    launch(dma, LOAD)
    for _ in range(4):
        tick(dma)                                   # through T+4
    event = tick(dma, lsu_banks={0})                # T+5: data for beat 0, bank 0 busy
    assert event.vmem_write == [0, 0, False] and event.d is None
    assert dma.backend.outstanding == 2
    event = tick(dma, lsu_banks={0})                # T+6: still held; beat 1 waits behind it
    assert event.vmem_write == [0, 0, False] and event.d is None
    event = tick(dma)                               # T+7: granted
    assert event.vmem_write == [0, 0, True] and event.d == 0
    assert tick(dma).d == 1


def test_lsu_access_to_another_bank_does_not_deny(dma):
    launch(dma, LOAD)
    for _ in range(4):
        tick(dma)
    assert tick(dma, lsu_banks={1, 2}).vmem_write == [0, 0, True]


def test_lsu_access_denies_the_store_path_vmem_read(dma):
    launch(dma, STORE)
    event = tick(dma, lsu_banks={0})                # T+1: denied, sequencer holds beat 0
    assert event.vmem_read == [0, 0, False]
    event = tick(dma)                               # T+2: beat 0 granted
    assert event.vmem_read == [0, 0, True]
    event = tick(dma)                               # T+3: beat 1 granted, no data yet for A
    assert event.vmem_read == [0, 1, True] and event.a is None
    assert tick(dma).a == [0, 0x2000, True]         # T+4: beat 0 requested


def test_dma_write_beats_dma_read_to_the_same_bank(dma):
    """Vmem.scala: DMA write > DMA read. A 1 KiB store behind a load reads bank 0 too."""
    launch(dma, LOAD)                                           # 1: load slot active from 2
    launch(dma, DMA_STORE_CH1(rd=x(6), rs1=x(1), rs2=x(3)))     # 2: load Get 0
    tick(dma)                                                   # 3: load Get 1; load dispatched at the edge
    event = tick(dma)                                           # 4: store is now the request head
    assert event.vmem_read == [0, 0, True]
    assert tick(dma).vmem_read == [0, 1, True]                  # 5
    event = tick(dma)                                           # 6: load data 0 (latency 4) and Put 0
    assert event.vmem_write == [0, 0, True] and event.a == [2, 0x2000, True]
    assert event.vmem_read == [0, 2, False], "write wins the bank 0 port"
    event = tick(dma)                                           # 7: load data 1, Put 1
    assert event.vmem_write == [0, 1, True] and event.a == [3, 0x2020, True]
    assert event.vmem_read == [0, 2, False]
    event = tick(dma)                                           # 8: read resumes; no data for Put 2 yet
    assert event.vmem_read == [0, 2, True] and event.a is None
    assert tick(dma).a is None                                  # 9
    assert tick(dma).a == [4, 0x2040, True]                     # 10: beat 2 two cycles after its grant


def test_commands_pipeline_across_channels(dma):
    """Beats of the next command issue while the first command's data is still returning."""
    launch(dma, DMA_LOAD_CH0(rd=x(1), rs1=x(2), rs2=x(3)))    # 1: 32 beats
    second = launch(dma, DMA_LOAD_CH1(rd=x(5), rs1=x(6), rs2=x(3)))    # 2: 32 beats, VMEM 1024; Get 0
    assert second.a == [0, 0x100, False]
    events = run(dma)
    fires = [event.cycle for event in events if event.a is not None]
    assert fires == list(range(3, 66)), "64 Gets back to back"
    retire = [event.cycle for event in events if event.d is not None][-1]
    assert retire == 65 + 4                                     # last Get at 65, latency 4
    assert dma.cycle == 69


def test_outstanding_beats_are_capped_at_max_in_flight(dma):
    dma.backend = FixedLatencyBackend(latency=200)
    launch(dma, DMA_LOAD_CH0(rd=x(1), rs1=x(2), rs2=x(3)))    # 32 beats
    launch(dma, DMA_LOAD_CH1(rd=x(5), rs1=x(6), rs2=x(3)))    # 32 beats
    launch(dma, DMA_LOAD_CH2(rd=x(7), rs1=x(6), rs2=x(3)))    # 3: 32 more, must wait
    while dma.cycle < 201:
        tick(dma)
    assert dma.backend.outstanding == 64
    assert dma._in_flight == 64
    assert dma.last_cycle.a is None                             # throttled
    # The first response frees a slot; the registered count allows a Get next cycle.
    event = tick(dma)                                           # 202 = 2 + 200: D for beat 0
    assert event.d == 0 and event.a is None
    assert tick(dma).a == [0, 0x2000, False]                    # ID 0 wrapped and free again


def test_responses_out_of_order_retire_slots_out_of_order(dma):
    """A fast second command retires before a slow first one (per-slot counters)."""
    dma.backend = FixedLatencyBackend(latency=4, latency_fn=lambda r: 40 if r.address < 0x1000 else 4)
    launch(dma, LOAD)                                           # two slow beats at 0x100
    launch(dma, DMA_LOAD_CH1(rd=x(5), rs1=x(6), rs2=x(4)))      # two fast beats at 0x2000
    state = dma.arch_state
    for _ in range(8):
        tick(dma)
    assert not state.check_flag(1), "channel 1 retired first"
    assert state.check_flag(0)
    assert dma.in_flight[0].insn.mnemonic == "dma.load.ch0"
    run(dma)
    assert not state.check_flag(0)


def test_source_id_in_flight_blocks_its_reuse(dma):
    """idInFlight: a wrapped source ID waits for the earlier beat with that ID."""
    dma.num_ids = 2                                             # small ID space for the test
    dma.max_in_flight = 2
    dma.backend = FixedLatencyBackend(latency=4, latency_fn=lambda r: 20 if r.source == 0 else 1)
    launch(dma, DMA_LOAD_CH0(rd=x(1), rs1=x(2), rs2=x(3)))
    fires = []
    for _ in range(12):
        event = tick(dma)
        if event.a is not None:
            fires.append((event.cycle, event.a[0]))
    # IDs 0 and 1 at cycles 2 and 3; ID 1 returns at 5, so ID 1 would be free
    # for cycle 6, but the sequencer needs ID 0 next, which is busy until 22.
    assert fires == [(2, 0), (3, 1)]


def test_backend_backpressure_stalls_channel_a(dma):
    dma.backend = FixedLatencyBackend(latency=4, accept_fn=lambda cycle: cycle % 3 != 0)
    launch(dma, DMA_LOAD_CH0(rd=x(1), rs1=x(2), rs2=x(3)))
    fires = [event.cycle for event in run(dma) if event.a is not None]
    assert len(fires) == 32 and not any(cycle % 3 == 0 for cycle in fires)


def test_link_rate_limits_beat_acceptance(dma):
    dma.backend = FixedLatencyBackend(latency=4, cycles_per_beat=16)
    launch(dma, DMA_LOAD_CH0(rd=x(1), rs1=x(2), rs2=x(3)))
    fires = [event.cycle for event in run(dma) if event.a is not None]
    assert fires == [2 + 16 * i for i in range(32)]
    assert dma.cycle == fires[-1] + 4


def test_dma_config_updates_the_base_at_issue_without_a_slot(dma):
    state = dma.arch_state
    state.base = 1
    dma.tick(Uop(DMA_CONFIG_CH0(rs1=x(1))))                     # T: base := x1 = 0, no slot, no flag
    assert state.base == 0 and not dma.has_in_flight and dma.last_cycle.busy == [False] * 8
    launch(dma, LOAD)                                           # T+1 launch already sees the new base
    assert dma._slots[0].dram_addr == 0x100
    state.write_xrf(8, 1)
    dma.tick(Uop(DMA_CONFIG_CH0(rs1=x(8))))                     # Not queued behind the in-flight load.
    assert state.base == 1 and len(dma._slots) == 1


def test_default_backend_is_the_vcs_measured_curve(dma):
    cfg = DefaultHardwareConfig()
    unit = DmaExecutionUnit("DMA0", Mock(spec=Logger), dma.arch_state, config=cfg)
    assert isinstance(unit.backend, CurveMemoryBackend)
    assert unit.backend.path.endswith("memory_curves/ee290sim_vcs_probe.json")
    assert unit.backend.latency_mode == "unloaded"
    assert (unit.backend.latency_at(0.0, 100), unit.backend.latency_at(0.0, 0)) == (46.0, 42.0)
    # Lone 64 B load: the Get at T+1 pays the load curve's 46-cycle lead-off
    # (D at T+47); the Get at T+2 is paced one beat time (1 / peak bandwidth,
    # about 23.5 cycles) behind it, so D at ceil(T+70.5) = T+71.
    launch_record = launch(unit, LOAD)
    launch_cycle = launch_record.cycle
    records = run(unit, 100)
    d_cycles = [record.cycle for record in records if record.d is not None]
    assert d_cycles == [launch_cycle + 47, launch_cycle + 71]


def test_zero_and_unaligned_sizes_are_rejected(dma):
    dma.arch_state.write_xrf(9, 0)
    with pytest.raises(RuntimeError, match="multiples of 32"):
        launch(dma, DMA_LOAD_CH0(rd=x(1), rs1=x(2), rs2=x(9)))
    dma.arch_state.write_xrf(9, 48)
    with pytest.raises(RuntimeError, match="multiples of 32"):
        launch(dma, DMA_LOAD_CH1(rd=x(1), rs1=x(2), rs2=x(9)))


def test_busy_channel_reissue_is_rejected(dma):
    launch(dma, LOAD)
    with pytest.raises(RuntimeError, match="channel 0 is busy"):
        dma.tick(Uop(DMA_LOAD_CH0(rd=x(5), rs1=x(2), rs2=x(4))))


def test_word_addressed_vmem_operands_follow_atlascore(dma):
    state = dma.arch_state
    state.write_xrf(9, 256)                                     # word 256 = byte 1024
    expected = state.dram[0x100:0x140].clone()
    launch(dma, DMA_LOAD_CH0(rd=x(9), rs1=x(2), rs2=x(4)))
    assert dma._slots[0].vmem_addr == 1024
    run(dma)
    assert torch.equal(state.vmem[1024:1088], expected)


def test_vmem_operand_low_word_bits_and_high_bits_are_dropped(dma):
    # vmemLineAddr = vmemAddr(wordAddrBits-1, wordOffBits): bits [18:3] for 1.5 MiB.
    state = dma.arch_state
    state.write_xrf(9, 256 + 5)                                 # not line aligned: same line
    launch(dma, DMA_LOAD_CH0(rd=x(9), rs1=x(2), rs2=x(4)))
    assert dma._slots[0].vmem_addr == 1024
    dma.abort()
    state.write_xrf(9, (1 << 19) + 256)                         # bit 19 is above wordAddrBits
    launch(dma, DMA_LOAD_CH0(rd=x(9), rs1=x(2), rs2=x(4)))
    assert dma._slots[0].vmem_addr == 1024
    dma.abort()
    state.write_xrf(9, (393216 - 8) + 8)                        # one line past the last
    with pytest.raises(RuntimeError, match="exceeds VMEM capacity"):
        launch(dma, DMA_LOAD_CH0(rd=x(9), rs1=x(2), rs2=x(4)))
