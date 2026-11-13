"""
Structural-hazard checks for the tensor register file (MRF) and VMEM ports.

Software schedules Atlas: there are no interlocks, so an instruction issued
before the resources it needs are free is a program error, not a stall. The
performance model raises BankConflictError when that happens:
  - MRF: ScalarCore's logical register reservations (``reserve_mreg``) and the
    physical 1R1W banks of MregFile.scala (``access_mreg``), where mN and
    m(N + banks) share one bank.
  - VMEM: the physical bank each LSU access holds on a cycle, published one
    cycle ahead so grant-based clients (the DMA) can arbitrate against it.
"""

from npu_model.geometry import DEFAULT, Geometry


class BankConflictError(RuntimeError):
    """
    Raised when two concurrently executing instructions access the same
    SRAM bank in the tensor register file or VMEM.
    """


class BankConflictChecker:
    """Per-cycle MRF and VMEM port bookkeeping shared by every execution unit."""

    def __init__(self, geometry: Geometry = DEFAULT) -> None:
        self.geometry = geometry
        self._mreg_reservations: dict[str, tuple[frozenset[int], frozenset[int]]] = {}
        self._mreg_ports: dict[tuple[bool, int], tuple[int, int, str]] = {}
        self._mreg_cycle: int | None = None
        self._mreg_releases: list[tuple[str, bool, bool]] = []
        self._vmem_ports: tuple[int, dict[int, str]] = (0, {})

    def reset(self) -> None:
        self._mreg_reservations.clear()
        self._mreg_ports.clear()
        self._mreg_releases.clear()
        self._mreg_cycle = None
        self._vmem_ports = (0, {})

    # ------------------------------------------------------------------
    # VMEM physical banks (Vmem.scala arbitration)
    # ------------------------------------------------------------------

    def announce_vmem_ports(self, cycle: int, banks: dict[int, str]) -> None:
        """Declare the physical VMEM banks the LSU will access on ``cycle``.

        LSU accesses are always granted and are fully determined by LSU state
        before the cycle, so the LSU publishes them one cycle ahead. Units with
        grant-based ports (the DMA) consult ``vmem_bank_owner`` in that cycle,
        whatever the Python tick order.
        """
        self._vmem_ports = (cycle, dict(banks))

    def vmem_bank_owner(self, cycle: int, bank: int) -> str | None:
        """The LSU access holding ``bank`` on ``cycle``, or None if the bank is free."""
        announced, banks = self._vmem_ports
        if announced != cycle:
            return None
        return banks.get(bank)

    def begin_cycle(self, cycle: int) -> None:
        """Apply last edge's releases before this cycle's issue checks.

        Deferral makes issue legality independent of the Python unit tick order.
        Every unit can call this with the same cycle; it advances only once.
        """
        if cycle == self._mreg_cycle:
            return
        for owner, release_reads, release_writes in self._mreg_releases:
            reads, writes = self._mreg_reservations.get(owner, (frozenset(), frozenset()))
            reads = frozenset() if release_reads else reads
            writes = frozenset() if release_writes else writes
            if reads or writes:
                self._mreg_reservations[owner] = (reads, writes)
            else:
                self._mreg_reservations.pop(owner, None)
        self._mreg_releases.clear()
        self._mreg_ports.clear()
        self._mreg_cycle = cycle

    def reserve_mreg(self, owner: str, reads: frozenset[int],
                     writes: frozenset[int], *, allow_write_during_read: bool = False) -> None:
        """ScalarCore's direction-aware logical-register issue assertions.

        VLOAD deliberately checks pending writes only. Physical SRAM port
        conflicts are checked separately on actual read/write request cycles.
        """
        count = self.geometry.num_m_registers
        if any(reg < 0 or reg >= count for reg in reads | writes):
            raise BankConflictError(f"MRF register outside m0..m{count - 1}")
        for other, (busy_reads, busy_writes) in self._mreg_reservations.items():
            if other == owner:
                continue
            conflict = (reads & busy_writes) | (writes & busy_writes)
            if not allow_write_during_read:
                conflict |= writes & busy_reads
            if conflict:
                raise BankConflictError(
                    f"MRF bank conflict: '{owner}' accesses register(s) {sorted(conflict)} held by '{other}'"
                )
        self._mreg_reservations[owner] = (reads, writes)

    def release_mreg(self, owner: str, *, reads: bool = True, writes: bool = True) -> None:
        self._mreg_releases.append((owner, reads, writes))

    def access_mreg(self, cycle: int, reg: int, row: int, write: bool, owner: str) -> None:
        """Check the physical 1R1W banks in MregFile.scala.

        mN and m(N + banks) share a bank. One read and one write may coexist,
        except same-row read/write has undefined SyncReadMem data.
        """
        self.begin_cycle(cycle)
        bank = self.geometry.mreg_bank(reg)
        physical_row = self.geometry.physical_row(reg, row)
        key = (write, bank)
        if key in self._mreg_ports:
            other = self._mreg_ports[key][2]
            direction = "write" if write else "read"
            raise BankConflictError(
                f"MRF bank conflict: multiple {direction} ports targeting physical bank {bank}: {other}, {owner}"
            )
        opposite = self._mreg_ports.get((not write, bank))
        if opposite is not None and opposite[1] == physical_row:
            raise BankConflictError(
                f"Undefined MRF same-row read/write collision on physical bank {bank}, row {physical_row}"
            )
        self._mreg_ports[key] = (reg, physical_row, owner)
