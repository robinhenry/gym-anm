import gym

def run_env(id):
    env = gym.make(id)
    env.reset()

    for i in range(1000000):
        env.render()
        _, reward, _, _ = env.step(env.action_space.sample())
        # print(reward)

    env.close()

if __name__ == '__main__':
    ids = ['Acrobot-v1']

    for id in ids:
        run_env(id)