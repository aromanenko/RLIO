# ----------------------------------------------------------------------
# ------------------------------- ИМПОРТЫ ------------------------------
# ----------------------------------------------------------------------

import gym
import gym_rlio
import numpy as np

from tqdm import tqdm

# ----------------------------------------------------------------------
# --------------------------------- КОД --------------------------------
# ----------------------------------------------------------------------

print('1 - Инициализация среды')

env = gym.make('rlio-v0')

LOAD_ID = {
    4600: [
        555800, 616400, 564900, 582700, 404500, 589400, 1518900, 582800,
        835000, 587400, 617400, 819800, 1843100, 631500, 7562300, 11637400,
        3539700, 3540400, 12906800, 4095600, 886100, 4212800, 706600, 9339400,
        589700, 625700, 560100, 559800, 490400, 1617800, 744200, 720500,
        4285500, 615200, 1453400, 4043300, 571300, 808700, 101300, 6783400
    ]
}

env.load_data( products_dict=LOAD_ID )

print('2 - Генерируем все возможные политики для стратегии Constant Stock Policy (s-1, S)')

policies = []
max_sales = int( env.stores_data.demand.max() )

for i in range(1, max_sales + 1):
    policies.append( (i-1, i) )

print(f'\tПолучено {len(policies)} возможных политик')

print('3 - Отбираем политики, подходящие нам из условия на mply_qty')

actions = dict()
max_actions_space_size = -1

for store in LOAD_ID.keys():
    actions[store] = dict()
    for item in LOAD_ID[store]:
        actions[store][item] = list()
        _maxDemand = int(
            max(
                env.stores_data[(env.stores_data.store_id == store) & (env.stores_data.product_id == item)].s_qty.max(),
                env.stores_data[(env.stores_data.store_id == store) & (env.stores_data.product_id == item)].demand.max()
            )
        )
        _maxMplyQty = int(
            env.stores_data[(env.stores_data.store_id == store) & (env.stores_data.product_id == item)].mply_qty.max()
        )
        for a in policies:
            if (a[0] >= _maxMplyQty) and (a[1] <= max_sales + 1):
                actions[store][item].append(a)
        if len(actions[store][item]) > max_actions_space_size:
            max_actions_space_size = len(actions[store][item])

print(f'\tПолучено {max_actions_space_size} возможных политик')

print('4 - Приводим политики в формат action нашей среды')

act = []

for i in range(max_actions_space_size):
    for store in LOAD_ID.keys():
        for item in LOAD_ID[store]:
            if i < len(actions[store][item]):
                if len(act) <= i:
                    act.append( dict() )
                if store not in act[i].keys():
                    act[i][store] = dict()
                act[i][store][item] = actions[store][item][i]

# print('\n--- DEBUG ----\n')
# print(act)
# print('\n--------------\n')
# for i in range( len(act) ):
#     print(f'{i+1}\t{len(act[i][4600])}\t{",".join( map(str, act[i][4600].keys()) )}')
# print('\n--------------\n')

print('5 - Проход всех полученных политик')

reward_log = dict()

for store in LOAD_ID.keys():
    reward_log[store] = dict()
    for item in LOAD_ID[store]:
        reward_log[store][item] = []

for a in tqdm(act):

    local_env = gym.make('rlio-v0')

    local_load = dict()
    for store in a.keys():
        local_load[store] = list()
        for item in a[store].keys():
            local_load[store].append(item)
    local_env.load_data(products_dict=local_load)

    obs, rev, done, _ = local_env.reset()

    pbar = tqdm()
    while not done:
        obs, rev, done, _ = local_env.step( a )
        pbar.update(1)
    pbar.close()

    for store in local_load.keys():
        for item in local_load[store]:
            reward_log[store][item].append( sum(local_env.environment_data[store][item]['reward_log']) )

print('6 - Результаты:')

print('\tSTORE\tITEM\tMAX REWARD\tBEST POLICY')

for store in LOAD_ID.keys():
    for item in LOAD_ID[store]:
        print(f'\t{store}\t{item}\t{max(reward_log[store][item])}\t{actions[store][item][np.argmax(reward_log[store][item])]}')
