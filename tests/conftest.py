import pytest

import npu_model.configs.hardware as hardware_configs
from tests.helpers import cleanup_tracked_simulations


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--max-cycles",
        action="store",
        type=int,
        default=100000,
        help="Maximum cycles to run per simulation-heavy test.",
    )
    parser.addoption(
        "--hardware-config",
        action="store",
        default="DefaultHardwareConfig",
        help="Hardware configuration class to use for program execution tests.",
    )
    parser.addoption(
        "--sim-verbose",
        action="store_true",
        default=False,
        help="Show simulator stdout during tests.",
    )


@pytest.fixture
def max_cycles(pytestconfig: pytest.Config) -> int:
    return pytestconfig.getoption("max_cycles")


@pytest.fixture
def sim_verbose(pytestconfig: pytest.Config) -> bool:
    return pytestconfig.getoption("sim_verbose")


@pytest.fixture
def hardware_config_cls(pytestconfig: pytest.Config):
    config_name = pytestconfig.getoption("hardware_config")
    try:
        return getattr(hardware_configs, config_name)
    except AttributeError as exc:
        available = ", ".join(getattr(hardware_configs, "__all__", []))
        raise pytest.UsageError(
            f"Unknown hardware config '{config_name}'. Available: {available}"
        ) from exc


@pytest.fixture(autouse=True)
def cleanup_simulations_after_test():
    yield
    cleanup_tracked_simulations()
