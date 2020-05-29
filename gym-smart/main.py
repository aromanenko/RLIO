import gym
import gym_smart

env = gym.make('SmartEnv-v0')
env.load_data('/Users/fritzwilliams/Desktop/A Reinforcement Learning Approach for Inventory Optimization in Retail/MERGE_TABLE_STORE_4600.csv',
              '/Users/fritzwilliams/Desktop/A Reinforcement Learning Approach for Inventory Optimization in Retail/echelon_1_sl.csv',
              '/Users/fritzwilliams/Desktop/A Reinforcement Learning Approach for Inventory Optimization in Retail/demand_data.csv'
              )
env.load_pairs('/Users/fritzwilliams/Desktop/A Reinforcement Learning Approach for Inventory Optimization in Retail/not_empty_states.csv')

# state = env.initial_state(4600, 25100)
# print(state)

# pairs_data, _ = env.calculate_states()
# print(pairs_data.iloc[:2])

# demand = env.initial_demand(4600, 25100, 1)
# print(demand)

# demand_data = env.calculate_demand(1)
# print(demand_data.iloc[:2])

# env.step()
# env.reset()
