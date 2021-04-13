import gym
import gym_rlio

env = gym.make('rlio-v0')

env.load_data(
    products_dict={
        4600: [
            555800, 616400, 564900, 582700, 404500, 589400, 582800, 1518900,
            835000, 587400, 617400, 819800, 1843100, 631500, 7562300, 11637400,
            3539700, 3540400, 12906800, 4095600, 886100, 4212800, 706600, 9339400,
            589700, 625700, 560100, 559800, 490400, 1617800, 744200, 720500,
            4285500, 615200, 1453400, 4043300, 571300, 808700, 101300, 6783400
        ]
    }
)

print( len(env.action_space) )
print('---')
actions = []
for a in env.action_space:
    if a[1] <= 20:
        actions.append(a)
print( len(actions) )
print(actions)
exit()

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
