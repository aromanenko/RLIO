from gym.envs.registration import register

register(
    id='rlio-v0',
    entry_point='gym_rlio.envs:RlioBasicEnv',
)
