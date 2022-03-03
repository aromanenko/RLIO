# ----------------------------------------------------------------------
# ------------------------------- ИМПОРТЫ ------------------------------
# ----------------------------------------------------------------------

import gym
import gym_rlio

# ----------------------------------------------------------------------
# --------------------------------- КОД --------------------------------
# ----------------------------------------------------------------------

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

print(env.compute_baseline_realworld())
print('\n---\n')

# ----------------------------------------------------------------------

print(env.stores_data.head(10))
print('\n---\n')

# ----------------------------------------------------------------------

print(
    env.stores_data.groupby(['store_id', 'product_id']).agg(
        {
            'mply_qty': ['min', 'max']
        }
    ).reset_index()
)
print('\n---\n')

# ----------------------------------------------------------------------

print(
    env.stores_data.groupby(['store_id', 'product_id']).agg(
        {
            'batch_size': ['min', 'max']
        }
    ).reset_index()
)
print('\n---\n')

# ----------------------------------------------------------------------

print( f'Full descrete action space of size\t{len(env.action_space)}' )

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
        for a in env.action_space:
            if (a[0] >= _maxMplyQty) and (a[1] <= _maxDemand + 1):
                actions[store][item].append(a)
        if len(actions[store][item]) > max_actions_space_size:
            max_actions_space_size = len(actions[store][item])

print( f'Smaller descrete action space (bussiness rules applied: ROL >= mply_qty and OUL <= max(demand) + 1) of maximum size\t{max_actions_space_size}' )
print(actions)
print('\n---\n')

# ----------------------------------------------------------------------

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

print(act)
print('\n---\n')

# ----------------------------------------------------------------------

for i in range( len(act) ):
    print(f'{i+1}\t{len(act[i][4600])}\t{",".join( map(str, act[i][4600].keys()) )}')
print('\n---\n')

# ----------------------------------------------------------------------

obs, rev, done, _ = env.reset()
env.render()

while not done:
    act = {}
    for store in obs.keys():
        act[store] = {}
        for product in obs[store].keys():
            act[store][product] = (1, 5)

    obs, rev, done, _ = env.step( act )

    env.render()

    exit()
