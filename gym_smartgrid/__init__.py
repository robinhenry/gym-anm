from gym.envs.registration import register

register(
    id='acgrid-v0',
    entry_point='gym_smartgrid.envs:FooEnv',
)