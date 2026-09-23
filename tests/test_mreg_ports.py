import pytest

from npu_model.hardware.bank_conflict import BankConflictChecker, BankConflictError


def test_logical_hazards_and_vload_write_exception():
    checker = BankConflictChecker()
    checker.begin_cycle(1)
    checker.reserve_mreg('reader', frozenset({0}), frozenset())
    checker.reserve_mreg('other-reader', frozenset({0}), frozenset())
    with pytest.raises(BankConflictError, match='MRF bank conflict'):
        checker.reserve_mreg('writer', frozenset(), frozenset({0}))
    checker.reserve_mreg('vload', frozenset(), frozenset({0}), allow_write_during_read=True)
    with pytest.raises(BankConflictError):
        checker.reserve_mreg('late-reader', frozenset({0}), frozenset())


def test_releases_visible_only_after_clock_edge():
    checker = BankConflictChecker()
    checker.begin_cycle(1)
    checker.reserve_mreg('producer', frozenset(), frozenset({2}))
    checker.release_mreg('producer')
    checker.begin_cycle(1)
    with pytest.raises(BankConflictError):
        checker.reserve_mreg('consumer', frozenset({2}), frozenset())
    checker.begin_cycle(2)
    checker.reserve_mreg('consumer', frozenset({2}), frozenset())


@pytest.mark.parametrize('write', [False, True])
def test_m0_and_m32_share_a_physical_port(write):
    checker = BankConflictChecker()
    checker.access_mreg(1, 0, 0, write, 'first-port')
    with pytest.raises(BankConflictError, match='physical bank 0'):
        checker.access_mreg(1, 32, 0, write, 'second-port')
    checker.access_mreg(2, 32, 0, write, 'next-cycle')


def test_physical_bank_has_independent_read_and_write_ports():
    checker = BankConflictChecker()
    checker.access_mreg(1, 0, 0, False, 'read')
    checker.access_mreg(1, 32, 0, True, 'write')
    checker.access_mreg(2, 0, 0, False, 'read')
    with pytest.raises(BankConflictError, match='same-row'):
        checker.access_mreg(2, 0, 0, True, 'write')
