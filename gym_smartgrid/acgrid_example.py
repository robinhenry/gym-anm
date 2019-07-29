from gym_smartgrid.envs.smartgrid_env import SmartGridEnv2

# Initialize the environment.
env = SmartGridEnv2()
env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, _, _ = env.step(action)
    print('Bus P: ', obs[0])
    print('Bus Q: ', obs[1])
    print('Branch current: ', obs[2])
    print('SoC: ', obs[3])

print('Total reward: ', env.total_reward)
