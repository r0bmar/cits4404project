import pytest

from ..gym_pairs_trading import PairsTradingEnv
from ..gym_pairs_trading.envs.trading_sim import Actions

@pytest.fixture
def load_env():
    env = PairsTradingEnv("AAPL","MSFT", size='compact')
    return env

def test_reset_environment(load_env):
    load_env.reset()

def test_first_step(load_env):
    load_env.reset()
    load_env.step(Actions.HOLD)
