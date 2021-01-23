import gym
import gym_io

import numpy as np
import pandas as pd
from tqdm import tqdm

ALPHA_STEP = 0.01
PROBA_STEP = 0.70

env = gym.make('IOEnv-v0')

observations, rewards, is_done, _ = env.init(
    '/Users/mgcrp/Desktop/RLIO_tmp/data/MERGE_TABLE_STORE_4600_with_sl.csv',
    sku=[
        18268300, 18500800, 1934800, 5559900, 5506400,
        6287400, 631500, 2011600, 4036900, 18064100,
        644700, 582700, 587800, 720600, 12842200,
        18983400, 18903700, 19013200, 8649100, 14832100,
        8505400, 562900, 581700, 11691900, 590800,
        18919200, 18762900, 14109400, 19816700, 18208300,
        8502900, 18206600, 626700, 1476600, 17882500,
        11772500, 15450300, 720400, 6415900, 624100],
    loc=[4600]
)

df_actions = pd.DataFrame(columns=['state_id', 'action', 'reward'])

if is_done: exit()

# Initialize actions
print('Initializing actions...')
for obs in tqdm(observations):
    for action in obs['actions']:
        if len(df_actions[(df_actions.state_id == obs['state_id']) & (df_actions.action == action)]) == 0:
            df_actions = df_actions.append({'state_id': obs['state_id'], 'action': action, 'reward': 0}, ignore_index=True)
print('Actions successfuly initialized!')

print(df_actions.dtypes)

m = 0 # time step

while not is_done:
    chosen_actions = dict()

    for obs in tqdm(observations):
        if m == 0:
            alpha = ALPHA_STEP
            probability = 1
        else:
            alpha = ALPHA_STEP / m
            probability = PROBA_STEP / m

        non_exploratory = np.random.binomial(n=1, p=1-probability)
        if non_exploratory:
            action_index = df_actions[df_actions['state_id'] == obs['state_id']].reward.idxmax()
        else:
            action_index = df_actions[df_actions['state_id'] == obs['state_id']].reward.idxmax()
            if len(action_data[action_data['state']==i]) > 1:
                action_index = np.random.choice(
                    df_actions[ (df_actions['state_id'] == obs['state_id']) & (df_actions.index != action_index) ].index,
                    1
                )[0]

        chosen_actions[obs['state_id']] = df_actions.loc[action_index, 'action']

    print(chosen_actions)
    break

print(df_actions)
