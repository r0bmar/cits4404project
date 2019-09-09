from gym.envs.registration import register

register(
    id='pairs_trading-v0',
    entry_point='gym_pairs_trading.envs:PairsTradingEnv',
)