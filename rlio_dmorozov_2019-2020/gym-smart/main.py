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
# demand_data = env.calculate_demand(1)

shop_id = 4600
product_id = 386400
max_steps = 10
alpha = 0.01
probability = 0.1
LT = 1
alpha_order = 0.3

# env.learn(2, max_steps, alpha, probability, LT, alpha_order)
env.learn_single(shop_id, product_id, max_steps, alpha, probability, LT, alpha_order)

state_data, action_data, env_data = env.reset()

sales = 5
stock = 15
service_level = 0.95
order = 0

obs = pd.DataFrame([[shop_id, product_id, sales, stock, service_level, order]], columns=['location', 'sku', 'sales', 'stock', 'sl', 'order'])
print('Initial State:\n', obs)

for i in range(5):
    action = env.predict(obs)
    obs, rew = env.step(obs, action, LT, alpha_order)
    env.render(action, obs, rew)

final_rew = env.reward(obs)
env.render(None, obs, final_rew)
