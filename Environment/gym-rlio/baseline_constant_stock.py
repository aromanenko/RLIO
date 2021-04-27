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
        555800, 616400, 564900, 582700, 404500, 589400, 582800, 1518900,
        835000, 587400, 617400, 819800, 1843100, 631500, 7562300, 11637400,
        3539700, 3540400, 12906800, 4095600, 886100, 4212800, 706600, 9339400,
        589700, 625700, 560100, 559800, 490400, 1617800, 744200, 720500,
        4285500, 615200, 1453400, 4043300, 571300, 808700, 101300, 6783400
    ]
}

env.load_data(
    products_dict=LOAD_ID
)

# ----------------------------------------------------------------------

print('2 - Генерируем все возможные политики для стратегии Constant Stock Policy (s-1, S)')

policies = []
max_sales = int( env.stores_data.demand.max() )

for i in range(1, max_sales + 1):
    policies.append( (i-1, i) )

print(f'\tПолучено {len(policies)} возможных политик')

# ----------------------------------------------------------------------

print('3 - Проход всех полученных политик')

reward_log = dict()
for store in LOAD_ID.keys():
    reward_log[store] = dict()
    for item in LOAD_ID[store]:
        reward_log[store][item] = []

for policy in tqdm(policies):

    obs, rev, done, _ = env.reset()

    pbar = tqdm()
    while not done:
        act = {}
        for store in obs.keys():
            act[store] = {}
            for product in obs[store].keys():
                act[store][product] = policy

        obs, rev, done, _ = env.step( act )
        pbar.update(1)
    pbar.close()

    for store in LOAD_ID.keys():
        for item in LOAD_ID[store]:
            reward_log[store][item].append( sum(env.environment_data[store][item]['reward_log']) )

# ----------------------------------------------------------------------

print('4 - Результаты:')

print('\tSTORE\tITEM\tMAX REWARD\tBEST POLICY')
for store in LOAD_ID.keys():
    for item in LOAD_ID[store]:
        print(f'\t{store}\t{item}\t{max(reward_log[store][item])}\t{policies[np.argmax(reward_log[store][item])]}')
