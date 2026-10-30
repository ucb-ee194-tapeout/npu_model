# Issue rules: when may the next tensor instruction issue?

`npu_model/scheduling/issue_rules.py` answers that question programmatically.

## The one rule

An instruction B may issue G cycles after an earlier instruction A when, for
every resource both of them touch, B's use comes after A's. Three kinds of
resource, three meanings of "after":

| Resource kind | Examples | Constraint |
| --- | --- | --- |
| Exclusive occupancy | MXU read ports P0/P1 and write ports W0/W1; the weight and accumulator push paths; each accumulator's single read port and single write port; each physical MREG bank's read port and write port | B's occupancy interval must not overlap A's: `G >= last_A - first_B + 1` |
| Row data flow | accumulator rows, weight-slot columns, MREG rows | read after write: `read >= write + 1`; write after read: `write >= read` (every unit samples reads before it applies writes, so the same cycle is fine) |
| Logical reservation | ScalarCore's MregBankTracker | B may not read or write a register A still has a pending write on, nor write one A still reads (VLOAD exempt): `G >= free_from_A` |

The gap G counts issue cycles; `DELAY N` between A and B gives `G = N + 2`,
and G = 1 is the next instruction with no delay. Isolation is simply G large
enough that nothing overlaps; nothing about an operation's own timing
changes when another is in flight. The only thing back-to-back issue changes
is which of A's windows B's windows land in.

## The table

Everything operation-specific is a table of windows (offsets from the issue
cycle T, 32-row geometry), transcribed from the sequencers:

| Operation | Occupies | Reads rows | Writes rows | Reservation free from |
| --- | --- | --- | --- | --- |
| weight push `wK <- mS` | one read port (P1 if free, else P0) and bank `S` read port T..T+31; weight path T+1..T+32 | `mS` row r at T+r | slot `K` column c at T+1+c | T+33 |
| acc push FP8 `accN <- mS` | one read port and bank read port T..T+31; acc path and `accN` write port T+1..T+32 | `mS` row r at T+r | `accN` row r at T+1+r | T+33 |
| acc push BF16 `accN <- mS,mS+1` | P0 and P1 and both bank read ports T..T+31; acc path and `accN` write port T+1..T+32 | both rows r at T+r | `accN` row r at T+1+r | T+33 |
| pop `mD <- accN` (BF16: `mD,mD+1`) | W0 (and W1), bank write port(s) T..T+31 / T+1..T+32; `accN` read port T..T+31 | `accN` row r at T+r | `mD` row r at T+1+r | T+33 |
| matmul `accN <- mS x wK` | P0 and bank `S` read port T..T+31; `accN` write port T+63..T+94 (MXU0) or T+3..T+34 (MXU1); one of 3 (MXU0) or 2 (MXU1) in-flight slots until the last write | `mS` row r at T+r; slot `K` column c from T+1+c to T+63+c (MXU0, PE(i,j) uses it for row r at T+1+r+i+j) or every column at T+1..T+32 (MXU1) | `accN` row r at T+63+r (MXU0) or T+3+r (MXU1) | T+33 |
| matmul.acc | as matmul, plus `accN` read port T..T+31 | plus `accN` row r at T+r | | |
| VPU binary/unary BF16 | bank read ports: `vs` T..T+31, `vs+1` T+32..T+63 (same for `vs2`); bank write ports `vd` T+2..T+33, `vd+1` T+34..T+65 | same cycles | same cycles | reads T+64, writes T+66 |
| VPU row reductions (`vred*.row`) | bank read ports `vs`, `vs+1` T..T+31; write ports `vd`, `vd+1` T+2..T+33 (`vredsum.row`: T+7..T+38) | | | reads T+32, writes T+34 / T+39 |
| VPU column reductions (`vredsum`, `vredmin`, `vredmax`) | reads `vs` T..T+31 and T+64..T+95, `vs+1` T+32..T+63 and T+96..T+127; writes `vd` T+66..T+97, `vd+1` T+98..T+129 | | | reads T+128, writes T+130 |
| `vpack.bf16.fp8` | reads `vs2` pair as a binary op; writes `vd` every other cycle T+3..T+65 | | | reads T+64, writes T+66 |
| `vunpack.fp8.bf16` | reads `vs2` T..T+31; writes `vd` T+3..T+34, `vd+1` T+35..T+66 | | | reads T+32, writes T+67 |
| `vli.all`, `vli.row` / `vli.col`, `vli.one` | writes `vd` T+1..T+32 and `vd+1` T+33..T+64 / `vd` only | | | T+65 / T+33 |
| `vtrpose.xlu` | bank `vs1` read port T+1..T+32; bank `vd` write port T+34..T+65; the XLU | | | reads T+34, writes T+66 |
| `vload` / `vstore` | bank write port T+3..T+34 / read port T+1..T+32; the vload or vstore path; the VMEM bank (256 KiB, one port) T+1..T+32 / T+3..T+34 | VMEM line r at T+1+r (vload) | VMEM line r at T+3+r (vstore) | T+35 |
| scalar load | the scalar-load path T..T+2; VMEM bank at T+1; the scalar writeback slot at T+3 | VMEM word at T+1 | `rd` at T+3, no bypass | |
| scalar store | VMEM bank at T+1 | | VMEM word at T+1 | |
| taken branch, `jal`, `jalr` | the delay slot at T+1: no control-flow instruction may issue there (branches are treated as taken) | | | |
| DMA transfer | its channel, until a `dma.wait` on that channel | | | |

The VPU adds its two issue slots and the issue-busy bits of VectorFSM: an
op is live while its age is below latency - 1; a new op is refused while any
live op exists and either both slots are live, or the new op or the live op
is a two-input or row-reduction op, or the two are the same op or share a
lane-box logic group ({vadd, vsub, vredsum.row}, {vexp, vexp2}, {vsin, vcos},
{vsquare, vcube}, {vmaximum, vredmax}, {vminimum, vredmin}, all vli). So
two singles from different groups overlap freely; everything else waits
latency - 1 of the earlier op.

Decisions in the sequencers that are gates rather than data flow are kept
as named resources so they stay visible:

| Gate | Behaviour | Data-flow bound it stands in for |
| --- | --- | --- |
| `acc_row0_gate` | a matmul on `accN` blocks every later command on `accN` (pop, push, matmul) until its row 0 has landed: T+64 on MXU0, T+4 on MXU1 | identical to read-after-write on row 0 |
| `wslot_drain` (MXU0) | a matmul on slot K blocks a weight push to K until T+63 (`wbufDrainInit = rows + cols - 2`) | 62 (write-after-read on column 0: the last PE reads column c at T+63+c, the push writes it at T_B+1+c) |
| `acc_pushgate` (MXU1) | an accumulator push to `accN` blocks a matmul, matmul.acc or FP8 push on `accN` until T+33 | 30 for a matmul (accumulator write port), 2 for matmul.acc / pop (read-after-write), 32 for a push (push path) |
| `vpu_*` | the issue-busy bits above | the register ports alone would allow much denser issue |
| `scalar_writeback` | a scalar load's response at T+3 may not coincide with a scalar instruction that writes a register (or `seli`) | none: it is a write-port conflict on the scalar register file |

Two rules are *value* hazards: neither the RTL nor the model detects them,
so the engine reports them with `enforced=False` and the sweep ignores them.
A scalar load has no bypass, so a consumer must issue at T+4 or later (at
T+3 it reads the old value). A VMEM line written by `vstore` or a scalar
store is read back correctly only after the write cycle.

`mN` and `m(N+32)` share one physical 1R1W bank, so the port rule applies to
`N mod 32`; because they occupy physical rows 0..31 and 32..63 a reader of one
and a writer of the other never meet on a row, and the same-row read/write
collision the model checks is unreachable between different registers.

One VPU-internal exception to the read-port rule: two VPU ops streaming the
same bank *in phase* (the same row on the same cycle) share one SRAM read,
so a column reduction's second pass over `vs` (T+64..T+95) may coincide with
another op's pass over the same register issued 64 cycles later. The model
encodes this (its VectorFSM read cache); it is worth confirming on RTL.

Legality is not monotone in the gap. A scalar store may slip in one cycle
after a `vstore` to the same bank (the vector store's first VMEM write is at
T+3) but not two; a scalar instruction writing a register may follow a
scalar load at any gap except 3. `gap_violations` answers for a specific
gap; `min_gap` gives the first legal one.
## What the rule says about back-to-back matmuls

Derived by `min_gap`, confirmed on the model (see the sweep):

| Situation (same MXU) | Gap | Binding resource |
| --- | --- | --- |
| matmul after matmul, different accumulator | 32 | read port P0 (and MXU0's 3-deep in-flight tracker, which also binds at exactly 32) |
| matmul.acc after matmul on the same accumulator | 64 (MXU0), 32 (MXU1) | row 0 read-after-write (MXU0); P0 (MXU1; the data bound is 4) |
| pop after matmul on the same accumulator | 64 (MXU0), 4 (MXU1 overwrite matmul), 32 (MXU1 matmul.acc) | row 0 read-after-write; accumulator read port |
| overwrite matmul after pop of the same accumulator | 1 | nothing shared (the matmul does not read the accumulator) |
| weight push after matmul on the same slot | 63 (MXU0), 32 (MXU1) | drain gate; slot gate |
| matmul after weight push to the same slot, different source register | 1 (MXU0), 32 (MXU1) | column c lands at T+1+c, the array reaches it at T+2+c; MXU1 reads the whole slot at once |
| any two commands that stream the same MREG bank | 32 | the bank's read port |
| accumulator push after matmul on the same accumulator | 94 (MXU0), 34 (MXU1) | the accumulator's single write port: the push's rows T+1..T+32 may not overlap the matmul's result rows |

So a scheduler needs exactly these facts to issue matmuls densely:

1. One matmul per 32 cycles per MXU, full stop (P0).
2. On MXU0, consecutive matmuls must alternate accumulators to sustain that
   rate, and a K-chain on one accumulator runs at 64. On MXU1 one accumulator
   sustains 32.
3. On MXU0, a slot can be re-pushed 63 cycles after the last matmul that used
   it, and the next matmul on it may follow the push by 1 cycle. Two slots
   alternating at a 32-cycle matmul stride give a push window of exactly one
   cycle (push at T+63, matmul at T+64), which is the `perf_mm_*` cadence
   (push, matmul, `DELAY 29`).
4. Pops must wait 64 (MXU0) or 4 (MXU1) after the matmul that produced the
   accumulator, and pops are 32 apart (W0).
5. Anything else is the register file: 32 between streams of one bank, and
   the reservation free-from cycle (33 MXU, 35 LSU, 66 VPU/XLU writes) before
   touching a register another engine is writing.

The last row of the table above is a hazard the RTL dispatcher does *not*
gate (`accReuseHazard` only checks row 0); issuing the push at 64 would trip
the `AccumulationBuffers` "compute write and load target the same buffer"
assertion. The model rejects it; software must space it.

## Using it

```python
from npu_model.scheduling import Op, min_gap, schedule, explain_schedule, op_from_instruction

gap, why = min_gap(Op("matmul", 0, mreg=0, acc=0, wslot=0),
                   Op("pop_fp8", 0, mreg=8, acc=0))        # 64, [row 0 read-after-write ...]

ops = [op_from_instruction(insn) for insn in program.instructions if insn.mnemonic != "delay"]
print(explain_schedule(ops))   # earliest cycle, DELAY to insert, and the binding rule per op
```

`Scheduler.check(op, at)` reports the rules a fixed placement breaks, which
is what a heuristic needs to score a candidate without running the model.
Legality is not always monotone in the gap (a single push takes P1 if free,
else P0), so `Scheduler.place` scans cycles rather than solving a formula.
`op_from_instruction(insn, xrf)` needs the scalar register values to resolve
LSU addresses to VMEM banks and lines; without them LSU ops carry no VMEM
constraints. A DMA transfer holds its channel until a `dma.wait` on it;
`Scheduler.place` raises if a second transfer is placed without one, and
treats issue cycles after a `dma.wait` as relative to its release (the
wait's length is the memory system's, see `rtl-timing.md`).
