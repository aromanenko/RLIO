import gym
import gym_smart

env = gym.make('SmartEnv-v0')
env.load_data('/Users/fritzwilliams/Google Диск/A Reinforcement Learning Approach for Inventory Optimization in Retail/MERGE_TABLE_STORE_4600/MERGE_TABLE_STORE_4600.csv',
              '/Users/fritzwilliams/Google Диск/A Reinforcement Learning Approach for Inventory Optimization in Retail/MERGE_TABLE_STORE_4600/data/echelon/echelon_1_sl.csv',
              '/Users/fritzwilliams/Google Диск/A Reinforcement Learning Approach for Inventory Optimization in Retail/MERGE_TABLE_STORE_4600/demand_data_train.csv'
              )
print('load_data DONE!')
env.initial_state(4600, 25100)
# env.step()
# env.reset()
