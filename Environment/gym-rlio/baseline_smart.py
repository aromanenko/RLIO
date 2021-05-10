# ----------------------------------------------------------------------
# ------------------------------- ИМПОРТЫ ------------------------------
# ----------------------------------------------------------------------

import gym
import gym_rlio

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# ------------------------------- ФУНКЦИИ ------------------------------
# ----------------------------------------------------------------------

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

# ----------------------------------------------------------------------
# --------------------------- ГИПЕРПАРАМЕТРЫ ---------------------------
# ----------------------------------------------------------------------

MAX_DEMAND_DELTA = 5

ALPHA = 0.01
MAX_EPOCHS = 3
PROBABILITY = 0.7

# ----------------------------------------------------------------------
# --------------------------------- КОД --------------------------------
# ----------------------------------------------------------------------

print('1 - Инициализация среды')

env = gym.make('rlio-v0')

LOAD_ID = {
    4600: [
        555800, 616400, 564900, 582700, 404500, 589400, 582800, 1518900,
        835000, 587400, 617400, 819800, 1843100, 631500, 7562300, 11637400,
        3539700, 3540400, 12906800, 4095600, 886100, 4212800, 706600, 9339400,
        589700, 625700, 560100, 559800, 490400, 1617800, 744200, 720500,
        4285500, 615200, 1453400, 4043300, 571300, 808700, 101300, 6783400
    ]
}

env.load_data( products_dict=LOAD_ID )

# ----------------------------------------------------------------------

print('2 - Инициализация таблицы с наградами за действие')

tmp = []
for store in LOAD_ID.keys():
    for item in LOAD_ID[store]:
        _minMplyQty = env.stores_data[(env.stores_data.store_id == store) & (env.stores_data.product_id == item)].mply_qty.min()
        _maxDemand = env.stores_data[(env.stores_data.store_id == store) & (env.stores_data.product_id == item)].demand.max()

        for action in env.action_space:
            if action[0] >= _minMplyQty and action[1] < _maxDemand + MAX_DEMAND_DELTA:
                tmp.append(
                    {
                        'store_id': store,
                        'product_id': item,
                        'action': action,
                        'R': 0
                    }
                )

df_actions = pd.DataFrame(tmp)
del tmp

# ----------------------------------------------------------------------

print('3 - Обучение')

current_step = 0
current_epoch = 0
flag_isFirst = True

time_step = dict()
reward_rate = dict()
cumulative_reward = dict()
maximization_step = dict()
for store in LOAD_ID.keys():
    time_step[store] = dict()
    reward_rate[store] = dict()
    cumulative_reward[store] = dict()
    maximization_step[store] = dict()
    for item in LOAD_ID[store]:
        time_step[store][item] = 0
        reward_rate[store][item] = 0
        cumulative_reward[store][item] = 0
        maximization_step[store][item] = 0

while current_epoch < MAX_EPOCHS:

    obs, rev, done, _ = env.reset()

    while not done:

        # ---

        action_type = dict()

        if flag_isFirst:
            for store in obs.keys():
                action_type[store] = dict()
                for item in obs[store].keys():
                    action_type[store][item] = False
            flag_isFirst = False
        else:
            for store in obs.keys():
                action_type[store] = dict()
                for item in obs[store].keys():

                    if time_step[store][item] > 0:
                        _probability = PROBABILITY / time_step[store][item]
                    else:
                        _probability = 1

                    _actionType = np.random.binomial(n=1, p=1-_probability)
                    action_type[store][item] = _actionType

        # ---

        action_index = dict()

        for store in obs.keys():
            action_index[store] = dict()
            for item in obs[store].keys():
                if action_type[store][item]:
                    _actionIndex = df_actions[(df_actions.store_id == store) & (df_actions.product_id == item)].R.idxmax()
                else:
                    _actionIndex = df_actions[(df_actions.store_id == store) & (df_actions.product_id == item)].R.idxmax()
                    if len(df_actions[(df_actions.store_id == store) & (df_actions.product_id == item)]) > 1:
                        _actionIndex = np.random.choice(
                            df_actions[
                                (df_actions.store_id == store) &
                                (df_actions.product_id == item) &
                                (df_actions.action.apply(lambda x: x[0] >= obs[store][item]['mply_qty']))
                            ].index,
                            1
                        )[0]
                action_index[store][item] = _actionIndex

        # ---

        action = dict()

        for store in obs.keys():
            action[store] = dict()
            for item in obs[store].keys():
                action[store][item] = df_actions.loc[action_index[store][item], 'action']

        # ---

        obs, rev, done, _ = env.step( action )

        # ---

        for store in rev.keys():
            for item in rev[store].keys():

                if time_step[store][item] > 0:
                    _alpha = ALPHA / time_step[store][item]
                else:
                    _alpha = ALPHA

                _reward = rev[store][item]
                _rewardRate = reward_rate[store][item]
                _currentR = df_actions.loc[action_index[store][item], 'R']

                df_actions.loc[action_index[store][item], 'R'] = (1 - _alpha) * _currentR + _alpha * (_reward - _rewardRate + df_actions.loc[action_index[store][item], 'R'].max())

                if action_type[store][item]:
                    cumulative_reward[store][item] += _reward
                    maximization_step[store][item] += 1
                    reward_rate[store][item] = cumulative_reward[store][item] / maximization_step[store][item]

                time_step[store][item] += 1

        current_step += 1

        # ---

        # print(f'>>> DEBUG: Step {current_step}')
        # print('action_type')
        # pretty(action_type)
        # print('---')
        #
        # print('action_index')
        # pretty(action_index)
        # print('---')
        #
        # print('action')
        # pretty(action)
        # print('---')
        #
        # print('rev')
        # pretty(rev)
        # print('---')
        #
        # print('cumulative_reward')
        # pretty(cumulative_reward)
        # print('---')
        #
        # print('maximization_step')
        # pretty(maximization_step)
        # print('---')
        #
        # print('reward_rate')
        # pretty(reward_rate)
        # print('---')
        #
        # print('time_step')
        # pretty(time_step)
        #
        # print('\t===\t')

    current_epoch += 1

    print(f'\nResults after epoch #{current_epoch}:')
    for store in LOAD_ID.keys():
        for item in LOAD_ID[store]:
            _bestActionId = df_actions[(df_actions.store_id == store) & (df_actions.product_id == item)].R.idxmax()
            print(f'\tStore: {store}\tItem: {item}\tCumulative Reward: {cumulative_reward[store][item]}\tSteps with maximization_step: {maximization_step[store][item]}\tBest action: {df_actions.loc[_bestActionId, "action"]}')

    df_actions.to_excel(f'experiment_{current_epoch}epochs_{current_step}steps.xlsx', index=False)
