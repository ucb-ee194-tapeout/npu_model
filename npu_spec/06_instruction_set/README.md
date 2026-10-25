# Penguin TPU (Atlas Accelerator) Instruction Green Card

Conventions used in the opcode tables:

- Bit positions use the RISC-V convention, with bit `31` the MSB and bit `0`
  the LSB.
- The instruction table of section 3 comes from `npu_model/isa/atlas_isa.toml`
  (`python -m npu_model.isa.gen card`). Do not edit it: change the table and
  generate it again.
- `Hex` is the opcode, then funct3 or funct2, then funct7 or the 12-bit
  immediate, in hex, when the instruction has these selector fields.
- A word `w` is the instruction if `(w & mask) == match`. The bits that are not
  in the mask are not decoded.
- `Configs` gives the configurations that have the unit (`all`: `default`,
  `ipt` and `ipt-nvfp4`). In the other configurations the instruction is illegal.
- Verilog-style descriptions use architectural state names such as `x[...]`,
  `m[...]`, `e[...]`, `pc`, `VMEM[...]`, and `mxuN.acc`.
- whenever an instruction semantic uses `{m[r], m[r+1]}`, the encoded register
  field names the low register of the pair, and encoding `r = 63` is illegal.

## 1. Register Set

The baseline scalar procedure-call convention follows standard `RV32I` ABI
practice for the `x` register file. No procedure-call ABI has been frozen yet
for `m`, `e`, or MXU-local state; this card uses the conservative convention
`Caller` for those architected state elements.

| Register | ABI Name | Description | Saver |
|-------|---|---|---|
| `pc`  | `pc` | Architectural program counter, stored as an instruction-word index | N/A |
| `x0`  | `zero` | Constant zero | Fixed |
| `x1`  | `ra` | Return address | Caller |
| `x2`  | `sp` | Stack pointer | Callee |
| `x3`  | `gp` | Global pointer | Fixed |
| `x4`  | `tp` | Thread pointer | Fixed |
| `x5`  | `t0` | Temporary register 0 | Caller |
| `x6`  | `t1` | Temporary register 1 | Caller |
| `x7`  | `t2` | Temporary register 2 | Caller |
| `x8`  | `s0/fp` | Saved register 0 / frame pointer | Callee |
| `x9`  | `s1` | Saved register 1 | Callee |
| `x10` | `a0` | Argument / return value 0 | Caller |
| `x11` | `a1` | Argument / return value 1 | Caller |
| `x12` | `a2` | Argument 2 | Caller |
| `x13` | `a3` | Argument 3 | Caller |
| `x14` | `a4` | Argument 4 | Caller |
| `x15` | `a5` | Argument 5 | Caller |
| `x16` | `a6` | Argument 6 | Caller |
| `x17` | `a7` | Argument 7 | Caller |
| `x18` | `s2` | Saved register 2 | Callee |
| `x19` | `s3` | Saved register 3 | Callee |
| `x20` | `s4` | Saved register 4 | Callee |
| `x21` | `s5` | Saved register 5 | Callee |
| `x22` | `s6` | Saved register 6 | Callee |
| `x23` | `s7` | Saved register 7 | Callee |
| `x24` | `s8` | Saved register 8 | Callee |
| `x25` | `s9` | Saved register 9 | Callee |
| `x26` | `s10` | Saved register 10 | Callee |
| `x27` | `s11` | Saved register 11 | Callee |
| `x28` | `t3` | Temporary register 3 | Caller |
| `x29` | `t4` | Temporary register 4 | Caller |
| `x30` | `t5` | Temporary register 5 | Caller |
| `x31` | `t6` | Temporary register 6 | Caller |
| `m0-m63` | `-` | Flat tensor register file, `1024` bytes per register | Caller |
| `e0-e31` | `-` | Whole-tensor scale register file, one `FP8_E8M0` exponent per register | Caller |
| `mxu0.w0` | `-` | MXU0 FP8 weight slot 0 | Caller |
| `mxu0.w1` | `-` | MXU0 FP8 weight slot 1 | Caller |
| `mxu1.w0` | `-` | MXU1 FP8 weight slot 0 | Caller |
| `mxu1.w1` | `-` | MXU1 FP8 weight slot 1 | Caller |
| `mxu0.acc0` | `-` | MXU0 local `32 x 32 BF16` accumulation buffer 0 | Caller |
| `mxu0.acc1` | `-` | MXU0 local `32 x 32 BF16` accumulation buffer 1 | Caller |
| `mxu1.acc0` | `-` | MXU1 local `32 x 32 BF16` accumulation buffer 0 | Caller |
| `mxu1.acc1` | `-` | MXU1 local `32 x 32 BF16` accumulation buffer 1 | Caller |
| `dma.base` | `-` | DMA base address | Caller |

## 2. Core Instruction Formats

### 2.1 Scalar RV32I-Compatible Formats
The following ASCII layout diagrams mirror the standard RISC-V bit positions
(RV32, MSB is bit `31`, LSB is bit `0`). `SB` below corresponds to the RISC-V `B`
format (branches), and `UJ` corresponds to the RISC-V `J` format (jumps).

<style>
  .rv32i-format-table th, .rv32i-format-table td {
    border: 1px solid #999;
    padding: 2px 6px;
    text-align: center;
  }
  /* Remove outer "top border" and "left border" only */
  .rv32i-format-table thead th {
    border-top: none;
    border-left: none;
    border-right: none;
    font-weight: 200;
  }
  .rv32i-format-table thead th:first-child,
  .rv32i-format-table tbody td:first-child {
    border-left: none;
    border-top: none;
    border-bottom: none;
    text-align: right;
  }
</style>

<table class="rv32i-format-table" style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr>
      <th></th>
      <th>
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>31</span>
          <span>25</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>24</span>
          <span>20</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>19</span>
          <span>15</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>14</span>
          <span>12</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>11</span>
          <span>7</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>6</span>
          <span>0</span>
        </div>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>R</td>
      <td>funct7</td>
      <td>rs2</td>
      <td>rs1</td>
      <td>funct3</td>
      <td>rd</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>I</td>
      <td colspan=2>imm[11:0]</td>
      <td>rs1</td>
      <td>funct3</td>
      <td>rd</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>S</td>
      <td>imm[11:5]</td>
      <td>rs2</td>
      <td>rs1</td>
      <td>funct3</td>
      <td>imm[4:0]</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>SB</td>
      <td>imm[12|10:5]</td>
      <td>rs2</td>
      <td>rs1</td>
      <td>funct3</td>
      <td>imm[4:1|11]</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>U</td>
      <td colspan="4" style="text-align:center;">imm[31:12]</td>
      <td>rd</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>UJ</td>
      <td colspan="4" style="text-align:center;">imm[20|10:1|11|19:12]</td>
      <td>rd</td>
      <td>opcode</td>
    </tr>
  </tbody>
</table>



### 2.5 Tensor operation

<table class="rv32i-format-table" style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr>
      <th></th>
      <th>
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>31</span>
          <span>25</span>
        </div>
      </th>
      <th>
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>24</span>
          <span>20</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>19</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>18</span>
          <span>16</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>15</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>14</span>
          <span>13</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>12</span>
          <span>7</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>6</span>
          <span>0</span>
        </div>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VLS</td>
      <td colspan=2>imm[11:0]</td>
      <td colspan=3>rs1</td>
      <td>f2</td>
      <td>vd</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>VR</td>
      <td>funct7</td>
      <td colspan=2>vs2</td>
      <td colspan=3>vs1</td>
      <td>vd</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>VI</td>
      <td colspan=4>imm[15:0]</td>
      <td colspan=2>f3</td>
      <td>vd</td>
      <td>opcode</td>
    </tr>
  </tbody>
</table>

Software may rely on the mnemonic repertoire and architectural semantics of
those instruction families, but not yet on a frozen bit-exact subformat.

## 3. Instructions

Rows are ordered by hex value.

<!-- BEGIN generated by `python -m npu_model.isa.gen card`: do not edit -->
| Assembly | Fmt | Hex | Match / mask | Unit | Configs | Operation |
|---|---|---|---|---|---|---|
| `lb x(rd), imm(x(rs1))` | I | `03/0` | `00000003` / `0000707f` | lsu | all | `x[rd] = {{24{VMEM[x[rs1] + imm][7]}}, VMEM[x[rs1] + imm]}` |
| `lh x(rd), imm(x(rs1))` | I | `03/1` | `00001003` / `0000707f` | lsu | all | `x[rd] = {{16{VMEM[x[rs1] + imm][15]}}, VMEM[x[rs1] + imm]}` |
| `lw x(rd), imm(x(rs1))` | I | `03/2` | `00002003` / `0000707f` | lsu | all | `x[rd] = VMEM[x[rs1] + imm]` |
| `lbu x(rd), imm(x(rs1))` | I | `03/4` | `00004003` / `0000707f` | lsu | all | `x[rd] = {24'b0, VMEM[x[rs1] + imm]}` |
| `lhu x(rd), imm(x(rs1))` | I | `03/5` | `00005003` / `0000707f` | lsu | all | `x[rd] = {16'b0, VMEM[x[rs1] + imm]}` |
| `seld e(rd), imm(x(rs1))` | I | `03/6` | `00006003` / `0000707f` | lsu | all | `e[rd] = byte 0 of the VMEM word that holds byte x[rs1] + imm` |
| `seli e(rd), imm` | I | `03/7` | `00007003` / `0000707f` | scalar | all | `e[rd] = imm[7:0]` |
| `vload m(vd), imm(x(rs1))` | VLS | `07/0` | `00000007` / `0000607f` | lsu | all | `m[vd] = VMEM[x[rs1] + (imm << 5)]` |
| `vstore m(vd), imm(x(rs1))` | VLS | `07/1` | `00002007` / `0000607f` | lsu | all | `VMEM[x[rs1] + (imm << 5)] = m[vd]` |
| `vload.scale m(vd), imm(x(rs1))` | VLS | `07/2` | `00004007` / `0000607f` | lsu | ipt-nvfp4 | `scales of m[vd] = 128 B of VMEM at x[rs1] + (imm << 5)` |
| `vstore.scale m(vd), imm(x(rs1))` | VLS | `07/3` | `00006007` / `0000607f` | lsu | ipt-nvfp4 | `128 B of VMEM at x[rs1] + (imm << 5) = scales of m[vd]` |
| `fence` | I | `0F/0` | `0000000f` / `0000707f` | scalar | all | `no operation` |
| `addi x(rd), x(rs1), imm` | I | `13/0` | `00000013` / `0000707f` | scalar | all | `x[rd] = x[rs1] + imm` |
| `slli x(rd), x(rs1), shamt` | I | `13/1/00` | `00001013` / `fe00707f` | scalar | all | `x[rd] = x[rs1] << shamt` |
| `slti x(rd), x(rs1), imm` | I | `13/2` | `00002013` / `0000707f` | scalar | all | `x[rd] = ($signed(x[rs1]) < $signed(imm))` |
| `sltiu x(rd), x(rs1), imm` | I | `13/3` | `00003013` / `0000707f` | scalar | all | `x[rd] = (x[rs1] < $unsigned($signed(imm)))` |
| `xori x(rd), x(rs1), imm` | I | `13/4` | `00004013` / `0000707f` | scalar | all | `x[rd] = x[rs1] ^ imm` |
| `srli x(rd), x(rs1), shamt` | I | `13/5/00` | `00005013` / `fe00707f` | scalar | all | `x[rd] = x[rs1] >> shamt` |
| `srai x(rd), x(rs1), shamt` | I | `13/5/20` | `40005013` / `fe00707f` | scalar | all | `x[rd] = $signed(x[rs1]) >>> shamt` |
| `ori x(rd), x(rs1), imm` | I | `13/6` | `00006013` / `0000707f` | scalar | all | `x[rd] = x[rs1] \| imm` |
| `andi x(rd), x(rs1), imm` | I | `13/7` | `00007013` / `0000707f` | scalar | all | `x[rd] = x[rs1] & imm` |
| `auipc x(rd), imm` | U | `17` | `00000017` / `0000007f` | scalar | all | `x[rd] = pc + {imm, 12'b0}` |
| `sb x(rs2), imm(x(rs1))` | S | `23/0` | `00000023` / `0000707f` | lsu | all | `VMEM[x[rs1] + imm] = x[rs2][7:0]` |
| `sh x(rs2), imm(x(rs1))` | S | `23/1` | `00001023` / `0000707f` | lsu | all | `VMEM[x[rs1] + imm] = x[rs2][15:0]` |
| `sw x(rs2), imm(x(rs1))` | S | `23/2` | `00002023` / `0000707f` | lsu | all | `VMEM[x[rs1] + imm] = x[rs2]` |
| `add x(rd), x(rs1), x(rs2)` | R | `33/0/00` | `00000033` / `fe00707f` | scalar | all | `x[rd] = x[rs1] + x[rs2]` |
| `sub x(rd), x(rs1), x(rs2)` | R | `33/0/20` | `40000033` / `fe00707f` | scalar | all | `x[rd] = x[rs1] - x[rs2]` |
| `sll x(rd), x(rs1), x(rs2)` | R | `33/1/00` | `00001033` / `fe00707f` | scalar | all | `x[rd] = x[rs1] << x[rs2][4:0]` |
| `slt x(rd), x(rs1), x(rs2)` | R | `33/2/00` | `00002033` / `fe00707f` | scalar | all | `x[rd] = ($signed(x[rs1]) < $signed(x[rs2]))` |
| `sltu x(rd), x(rs1), x(rs2)` | R | `33/3/00` | `00003033` / `fe00707f` | scalar | all | `x[rd] = (x[rs1] < x[rs2])` |
| `xor x(rd), x(rs1), x(rs2)` | R | `33/4/00` | `00004033` / `fe00707f` | scalar | all | `x[rd] = x[rs1] ^ x[rs2]` |
| `srl x(rd), x(rs1), x(rs2)` | R | `33/5/00` | `00005033` / `fe00707f` | scalar | all | `x[rd] = x[rs1] >> x[rs2][4:0]` |
| `sra x(rd), x(rs1), x(rs2)` | R | `33/5/20` | `40005033` / `fe00707f` | scalar | all | `x[rd] = $signed(x[rs1]) >>> x[rs2][4:0]` |
| `or x(rd), x(rs1), x(rs2)` | R | `33/6/00` | `00006033` / `fe00707f` | scalar | all | `x[rd] = x[rs1] \| x[rs2]` |
| `and x(rd), x(rs1), x(rs2)` | R | `33/7/00` | `00007033` / `fe00707f` | scalar | all | `x[rd] = x[rs1] & x[rs2]` |
| `lui x(rd), imm` | U | `37` | `00000037` / `0000007f` | scalar | all | `x[rd] = {imm, 12'b0}` |
| `vadd.bf16 m(vd), m(vs1), m(vs2)` | VR | `57/00` | `00000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = {m[vs1], m[vs1+1]} + {m[vs2], m[vs2+1]}` |
| `vredsum.bf16 m(vd), m(vs1)` | VR | `57/01` | `02000057` / `fe00007f` | vpu | all | `lane j: sum of column j of m[vs1] and m[vs1+1], in every row of m[vd] and m[vd+1]` |
| `vsub.bf16 m(vd), m(vs1), m(vs2)` | VR | `57/02` | `04000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = {m[vs1], m[vs1+1]} - {m[vs2], m[vs2+1]}` |
| `vmul.bf16 m(vd), m(vs1), m(vs2)` | VR | `57/03` | `06000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = {m[vs1], m[vs1+1]} * {m[vs2], m[vs2+1]}` |
| `vminimum.bf16 m(vd), m(vs1), m(vs2)` | VR | `57/04` | `08000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = min({m[vs1], m[vs1+1]}, {m[vs2], m[vs2+1]})` |
| `vredmin.bf16 m(vd), m(vs1)` | VR | `57/05` | `0a000057` / `fe00007f` | vpu | all | `lane j: min of column j of m[vs1] and m[vs1+1], in every row of m[vd] and m[vd+1]` |
| `vmaximum.bf16 m(vd), m(vs1), m(vs2)` | VR | `57/06` | `0c000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = max({m[vs1], m[vs1+1]}, {m[vs2], m[vs2+1]})` |
| `vredmax.bf16 m(vd), m(vs1)` | VR | `57/07` | `0e000057` / `fe00007f` | vpu | all | `lane j: max of column j of m[vs1] and m[vs1+1], in every row of m[vd] and m[vd+1]` |
| `vredsum.row.bf16 m(vd), m(vs1)` | VR | `57/21` | `42000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]}[i, j] = sum({m[vs1], m[vs1+1]}[i, :])` |
| `vredmin.row.bf16 m(vd), m(vs1)` | VR | `57/24` | `48000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]}[i, j] = min({m[vs1], m[vs1+1]}[i, :])` |
| `vredmax.row.bf16 m(vd), m(vs1)` | VR | `57/26` | `4c000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]}[i, j] = max({m[vs1], m[vs1+1]}[i, :])` |
| `vmov m(vd), m(vs1)` | VR | `57/40` | `80000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = {m[vs1], m[vs1+1]}` |
| `vrecip.bf16 m(vd), m(vs1)` | VR | `57/41` | `82000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = 1 / {m[vs1], m[vs1+1]}` |
| `vexp.bf16 m(vd), m(vs1)` | VR | `57/42` | `84000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = exp({m[vs1], m[vs1+1]})` |
| `vexp2.bf16 m(vd), m(vs1)` | VR | `57/43` | `86000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = 2^{m[vs1], m[vs1+1]}` |
| `vpack.bf16.fp8 m(vd), m(vs2), e(vs1)` | VR | `57/44` | `88000057` / `fe00007f` | vpu | all | `m[vd] = E4M3({m[vs2], m[vs2+1]} / 2^(e[vs1] - 127)) (FP8 row k = BF16 rows 2k, 2k+1)` |
| `vunpack.fp8.bf16 m(vd), m(vs2), e(vs1)` | VR | `57/45` | `8a000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = BF16(m[vs2] * 2^(e[vs1] - 127)) (vpack's row layout)` |
| `vsquare.bf16 m(vd), m(vs1)` | VR | `57/46` | `8c000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = {m[vs1], m[vs1+1]}^2` |
| `vcube.bf16 m(vd), m(vs1)` | VR | `57/47` | `8e000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = {m[vs1], m[vs1+1]}^3` |
| `vrelu.bf16 m(vd), m(vs1)` | VR | `57/48` | `90000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = max({m[vs1], m[vs1+1]}, 0)` |
| `vsin.bf16 m(vd), m(vs1)` | VR | `57/49` | `92000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = sin({m[vs1], m[vs1+1]})` |
| `vcos.bf16 m(vd), m(vs1)` | VR | `57/4A` | `94000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = cos({m[vs1], m[vs1+1]})` |
| `vtanh.bf16 m(vd), m(vs1)` | VR | `57/4B` | `96000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = tanh({m[vs1], m[vs1+1]})` |
| `vlog2.bf16 m(vd), m(vs1)` | VR | `57/4C` | `98000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = log2({m[vs1], m[vs1+1]})` |
| `vsqrt.bf16 m(vd), m(vs1)` | VR | `57/4D` | `9a000057` / `fe00007f` | vpu | all | `{m[vd], m[vd+1]} = sqrt({m[vs1], m[vs1+1]})` |
| `vli.all m(vd), imm` | VI | `5F/0` | `0000005f` / `0000e07f` | vpu | all | `{m[vd], m[vd+1]}[i, j] = imm` |
| `vli.row m(vd), imm` | VI | `5F/1` | `0000205f` / `0000e07f` | vpu | all | `{m[vd], m[vd+1]}[0, j] = imm; the other rows = 0` |
| `vli.col m(vd), imm` | VI | `5F/2` | `0000405f` / `0000e07f` | vpu | all | `m[vd][i, 0] = imm; the other elements = 0` |
| `vli.one m(vd), imm` | VI | `5F/3` | `0000605f` / `0000e07f` | vpu | all | `m[vd][0, 0] = imm; the other elements = 0` |
| `beq x(rs1), x(rs2), offset` | B | `63/0` | `00000063` / `0000707f` | scalar | all | `if (x[rs1] == x[rs2]) pc = pc + offset after 1 delay slot` |
| `bne x(rs1), x(rs2), offset` | B | `63/1` | `00001063` / `0000707f` | scalar | all | `if (x[rs1] != x[rs2]) pc = pc + offset after 1 delay slot` |
| `blt x(rs1), x(rs2), offset` | B | `63/4` | `00004063` / `0000707f` | scalar | all | `if ($signed(x[rs1]) < $signed(x[rs2])) pc = pc + offset after 1 delay slot` |
| `bge x(rs1), x(rs2), offset` | B | `63/5` | `00005063` / `0000707f` | scalar | all | `if ($signed(x[rs1]) >= $signed(x[rs2])) pc = pc + offset after 1 delay slot` |
| `bltu x(rs1), x(rs2), offset` | B | `63/6` | `00006063` / `0000707f` | scalar | all | `if (x[rs1] < x[rs2]) pc = pc + offset after 1 delay slot` |
| `bgeu x(rs1), x(rs2), offset` | B | `63/7` | `00007063` / `0000707f` | scalar | all | `if (x[rs1] >= x[rs2]) pc = pc + offset after 1 delay slot` |
| `jalr x(rd), x(rs1), imm` | I | `67/0` | `00000067` / `0000707f` | scalar | all | `next_pc = x[rs1] + imm; x[rd] = pc + 1; pc = next_pc after 1 delay slot` |
| `delay imm` | I | `67/1` | `00001067` / `0000707f` | scalar | all | `hold decode issue for imm cycles` |
| `vtrpose.xlu m(vd), m(vs1)` | VR | `6B/00` | `0000006b` / `fe00007f` | xlu | all | `m[vd] = m[vs1]^T (32 x 32 bytes)` |
| `jal x(rd), offset` | J | `6F` | `0000006f` / `0000007f` | scalar | all | `x[rd] = pc + 1; pc = pc + offset after 1 delay slot` |
| `ecall` | I | `73/0/000` | `00000073` / `ffffffff` | scalar | all | `halt_reason = ECALL; halt = 1'b1` |
| `ebreak` | I | `73/0/001` | `00100073` / `ffffffff` | scalar | all | `halt_reason = EBREAK; halt = 1'b1` |
| `csrrw x(rd), csr, x(rs1)` | I | `73/1` | `00001073` / `0000707f` | scalar | all | `x[rd] = CSR[csr]; CSR[csr] = x[rs1]` |
| `csrrs x(rd), csr, x(rs1)` | I | `73/2` | `00002073` / `0000707f` | scalar | all | `x[rd] = CSR[csr]; if (rs1 != 0) CSR[csr] = CSR[csr] \| x[rs1]` |
| `csrrc x(rd), csr, x(rs1)` | I | `73/3` | `00003073` / `0000707f` | scalar | all | `x[rd] = CSR[csr]; if (rs1 != 0) CSR[csr] = CSR[csr] & ~x[rs1]` |
| `csrrwi x(rd), csr, uimm` | I | `73/5` | `00005073` / `0000707f` | scalar | all | `x[rd] = CSR[csr]; CSR[csr] = uimm` |
| `csrrsi x(rd), csr, uimm` | I | `73/6` | `00006073` / `0000707f` | scalar | all | `x[rd] = CSR[csr]; if (uimm != 0) CSR[csr] = CSR[csr] \| uimm` |
| `csrrci x(rd), csr, uimm` | I | `73/7` | `00007073` / `0000707f` | scalar | all | `x[rd] = CSR[csr]; if (uimm != 0) CSR[csr] = CSR[csr] & ~uimm` |
| `vmatpush.weight.mxu0 w(vd), m(vs1)` | VR | `77/00` | `00000077` / `fe00007f` | mxu0 | default | `mxu0.w[vd] = m[vs1] (E4M3; row j = output column j)` |
| `vmatpush.weight.mxu1 w(vd), m(vs1)` | VR | `77/01` | `02000077` / `fe00007f` | mxu1 | all | `mxu1.w[vd] = m[vs1] (E4M3; row j = output column j)` |
| `vmatpush.acc.fp8.mxu0 acc(vd), m(vs1)` | VR | `77/02` | `04000077` / `fe00007f` | mxu0 | default | `mxu0.acc[vd] = BF16(m[vs1]) (E4M3)` |
| `vmatpush.acc.fp8.mxu1 acc(vd), m(vs1)` | VR | `77/03` | `06000077` / `fe00007f` | mxu1 | all | `mxu1.acc[vd] = BF16(m[vs1]) (E4M3)` |
| `vmatpush.acc.bf16.mxu0 acc(vd), m(vs1)` | VR | `77/04` | `08000077` / `fe00007f` | mxu0 | default | `mxu0.acc[vd] = {m[vs1], m[vs1+1]}` |
| `vmatpush.acc.bf16.mxu1 acc(vd), m(vs1)` | VR | `77/05` | `0a000077` / `fe00007f` | mxu1 | all | `mxu1.acc[vd] = {m[vs1], m[vs1+1]}` |
| `vmatpop.fp8.acc.mxu0 m(vd), acc(vs2), e(vs1)` | VR | `77/06` | `0c000077` / `fe00007f` | mxu0 | default | `m[vd] = E4M3(mxu0.acc[vs2] * 2^(e[vs1] - 127))` |
| `vmatpop.fp8.acc.mxu1 m(vd), acc(vs2), e(vs1)` | VR | `77/07` | `0e000077` / `fe00007f` | mxu1 | all | `m[vd] = E4M3(mxu1.acc[vs2] * 2^(e[vs1] - 127))` |
| `vmatpop.bf16.acc.mxu0 m(vd), acc(vs2)` | VR | `77/08` | `10000077` / `fe00007f` | mxu0 | default | `{m[vd], m[vd+1]} = mxu0.acc[vs2]` |
| `vmatpop.bf16.acc.mxu1 m(vd), acc(vs2)` | VR | `77/09` | `12000077` / `fe00007f` | mxu1 | all | `{m[vd], m[vd+1]} = mxu1.acc[vs2]` |
| `vmatmul.mxu0 acc(vd), m(vs1), w(vs2)` | VR | `77/0A` | `14000077` / `fe00007f` | mxu0 | default | `mxu0.acc[vd] = m[vs1] @ mxu0.w[vs2]^T` |
| `vmatmul.mxu1 acc(vd), m(vs1), w(vs2)` | VR | `77/0B` | `16000077` / `fe00007f` | mxu1 | all | `mxu1.acc[vd] = m[vs1] @ mxu1.w[vs2]^T` |
| `vmatmul.acc.mxu0 acc(vd), m(vs1), w(vs2)` | VR | `77/0C` | `18000077` / `fe00007f` | mxu0 | default | `mxu0.acc[vd] = mxu0.acc[vd] + m[vs1] @ mxu0.w[vs2]^T` |
| `vmatmul.acc.mxu1 acc(vd), m(vs1), w(vs2)` | VR | `77/0D` | `1a000077` / `fe00007f` | mxu1 | all | `mxu1.acc[vd] = mxu1.acc[vd] + m[vs1] @ mxu1.w[vs2]^T` |
| `vmatpush.weight.nv w(vd), m(vs1)` | VR | `77/0E` | `1c000077` / `fe00007f` | nv | ipt-nvfp4 | `nv.w[vd] = m[vs1] and its scales (row j = output column j)` |
| `vmatpush.acc.bf16.nv acc(vd), m(vs1)` | VR | `77/0F` | `1e000077` / `fe00007f` | nv | ipt-nvfp4 | `nv.acc[vd] = {m[vs1], m[vs1+1]} (BF16, exact in FP32)` |
| `vmatpush.acc.fp32.nv acc(vd), m(vs1)` | VR | `77/10` | `20000077` / `fe00007f` | nv | ipt-nvfp4 | `nv.acc[vd] = m[vs1..vs1+3] (register q holds columns 8q..8q+7)` |
| `vmatpop.bf16.acc.nv m(vd), acc(vs2)` | VR | `77/11` | `22000077` / `fe00007f` | nv | ipt-nvfp4 | `{m[vd], m[vd+1]} = BF16(nv.acc[vs2]) (round to nearest even)` |
| `vmatpop.fp32.acc.nv m(vd), acc(vs2)` | VR | `77/12` | `24000077` / `fe00007f` | nv | ipt-nvfp4 | `m[vd..vd+3] = nv.acc[vs2] (register q holds columns 8q..8q+7)` |
| `vmatmul.nv acc(vd), m(vs1), w(vs2)` | VR | `77/13` | `26000077` / `fe00007f` | nv | ipt-nvfp4 | `nv.acc[vd] = m[vs1] @ nv.w[vs2]^T (64 NVFP4 elements, FP32)` |
| `vmatmul.acc.nv acc(vd), m(vs1), w(vs2)` | VR | `77/14` | `28000077` / `fe00007f` | nv | ipt-nvfp4 | `nv.acc[vd] = nv.acc[vd] + m[vs1] @ nv.w[vs2]^T (one NVIDIA m16n8k64 step)` |
| `dma.load x(rd), x(rs1), x(rs2), ch` | R | `7B/00` | `0000007b` / `fe00007f` | dma | all | `channel ch: VMEM[x[rd]] = DRAM[{dma.base, x[rs1]}], x[rs2] bytes` |
| `dma.store x(rd), x(rs1), x(rs2), ch` | R | `7B/01` | `0200007b` / `fe00007f` | dma | all | `channel ch: DRAM[{dma.base, x[rd]}] = VMEM[x[rs1]], x[rs2] bytes` |
| `dma.config x(rs1), ch` | R | `7F/00` | `0000007f` / `fff00fff` | dma | all | `dma.base = x[rs1]` |
| `dma.wait ch` | R | `7F/01` | `0200007f` / `ffff8fff` | dma | all | `hold issue until channel ch is idle` |
<!-- END generated -->


DMA instructions take the channel as a trailing named operand, `ch0`-`ch7`,
encoded in `funct3`: `dma.load x(rd), x(rs1), x(rs2), ch<N>`,
`dma.store x(rd), x(rs1), x(rs2), ch<N>`, `dma.config x(rs1), ch<N>` and
`dma.wait ch<N>`. CSR instructions use the RISC-V operand order
`csrrw x(rd), csr, x(rs1)`; the immediate forms (`csrrwi`, `csrrsi`, `csrrci`)
take a 5-bit immediate in place of `rs1`.

## 4. Architectural Design Parameters

| Item | Value |
|---|---:|
| Instruction width | `32` bits |
| Instruction alignment | `4` bytes |
| Control-flow delay slots | `1` |
| Scalar registers | `32` |
| Tensor registers | `64` |
| Scale registers | `32` |
| Tensor register storage | `32 rows x 32 bytes = 1024 bytes` |
| MXU count | `2` |
| MXU weight slots per MXU | `2` |
| MXU accumulator storage | `32 x 32 BF16` |
| DMA channels | `8` |
| `IMEM` | `0x0002_0000` ~ `0x0004_0000` |
| `VMEM` | `0x2000_0000` ~ `0x2004_0000` |
| `PERIPH` (non-cacheable) | `0x4000_0000` ~ `0x8000_0000` |
| `DRAM` (cacheable) | `0x8000_0000` ~ `0x8_8000_0000` |
| `DONT TELL PROF` (non-cacheable) | `0x8_8000_0000` ~ `0x10_0000_0000` |

`PERIPH`, `DRAM`, and `DONT TELL PROF` are the off-chip memory region that is mapped on serial TileLink interface.
