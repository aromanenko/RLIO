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

# pairs_data, _ = env.calculate_states()
# print(pairs_data.iloc[:2])

# env.initial_demand(4600, 25100, 1)

# demand_data = env.calculate_demand(1)
# print(demand_data.iloc[:2])

# env.learn(2, 10, 0.01, 0.1, 1, 0.3)
env.learn_single(4600, 23100, 10, 0.01, 0.1, 1, 0.3)

state_data, action_data, env_data = env.reset()
# print('state_data:\n', state_data)
# print('\naction_data:\n', action_data)
# print('\nenv_data:\n', env_data)

obs = pd.DataFrame([[4600, 23100, 5, 10, 0.75, 0]], columns=['location', 'sku', 'sales', 'stock', 'sl', 'order'])
print('\nobs:\n', obs)

for i in range(1):
    action = env.predict(obs)
    print('\naction:\n', action)
    obs, reward = env.step(obs, action)
    print('\nobs:\n', obs)
    print('\nreward:\n', reward)
    # env.render()

# obs = env.reset()
# for i in range(2000):
#   action, _states = model.predict(obs)
#   obs, rewards, done, info = env.step(action)
#   env.render()
