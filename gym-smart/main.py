import gym
import gym_smart
import pandas as pd

env = gym.make('SmartEnv-v0')
env.load_data('/Users/fritzwilliams/Desktop/A Reinforcement Learning Approach for Inventory Optimization in Retail/MERGE_TABLE_STORE_4600.csv',
              '/Users/fritzwilliams/Desktop/A Reinforcement Learning Approach for Inventory Optimization in Retail/echelon_1_sl.csv',
              '/Users/fritzwilliams/Desktop/A Reinforcement Learning Approach for Inventory Optimization in Retail/demand_data.csv'
              )
env.load_pairs('/Users/fritzwilliams/Desktop/A Reinforcement Learning Approach for Inventory Optimization in Retail/not_empty_states.csv')

# env.initial_state(4600, 25100)
# env.initial_demand(4600, 25100, 1)

# pairs_data, _ = env.calculate_states()
# print(pairs_data.iloc[:2])

# demand_data = env.calculate_demand(1)
# print(demand_data.iloc[:2])

# env.learn(2, 10, 0.01, 0.1, 1, 0.3)
env.learn_single(4600, 386400, 10, 0.01, 0.1, 1, 0.3)

state_data, action_data, env_data = env.reset()

obs = pd.DataFrame([[4600, 386400, 5, 15, 0.95, 0]], columns=['location', 'sku', 'sales', 'stock', 'sl', 'order'])
print('Initial State:\n', obs)

for i in range(5):
    action = env.predict(obs)
    obs, rew = env.step(obs, action, 1, 0.3)
    env.render(action, obs, rew)

final_rew = env.reward(obs)
env.render(None, obs, final_rew)
