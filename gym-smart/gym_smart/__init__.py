from gym.envs.registration import register

register(id='SmartEnv-v0',
         entry_point='gym_smart.envs:SmartEnv'
)
