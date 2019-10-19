import pytest

from ..gym_pairs_trading.envs.data_source import DataSource

@pytest.fixture
def load_data_source():
    ds = DataSource("AAPL","MSFT", size='compact')
    return ds

def test_reset(load_data_source):
    load_data_source.reset()
