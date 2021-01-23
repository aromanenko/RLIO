from gym.envs.registration import register

register(
    id='IOEnv-v0',
    entry_point='gym_io.envs:IOEnv'
)
