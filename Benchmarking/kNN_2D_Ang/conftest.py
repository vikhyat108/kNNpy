import pytest

import numpy as np

def pytest_addoption(parser):
    parser.addoption("--NSIDE", action="store", default="256")
    parser.addoption("--n_tracer", action="store", default="20000")
    parser.addoption("--k_List", action="store", default="1")
    parser.addoption("--rounds", action="store", default="5")
    parser.addoption("--warmup", action="store", default="1")

@pytest.fixture(scope="session")
def NSIDE(pytestconfig):
    return int(pytestconfig.getoption("NSIDE"))

@pytest.fixture(scope="session")
def n_tracer(pytestconfig):
    return int(pytestconfig.getoption("n_tracer"))

@pytest.fixture(scope="session")
def k_List(pytestconfig):
    return list(np.array(pytestconfig.getoption("k_List").split()).astype(int))

@pytest.fixture(scope="session")
def rounds(pytestconfig):
    return int(pytestconfig.getoption("rounds"))

@pytest.fixture(scope="session")
def warmup(pytestconfig):
    return int(pytestconfig.getoption("warmup"))