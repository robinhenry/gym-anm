.. gym-anm quickstart documentation

Quickstart
==============

After installation, you should be able to train your agents on :code:`gym-anm` environments.

Existing environment
----------------------
If the task you would like to solve already exist, the environment can be initialized using
:code:`gym.make('gym_anm:<ENV_ID>')`. The following example uses the :ref:`ANM6Easy-v0 <anm6_label>` task and the actions
are randomly sampled from the action space at each time step: ::

    import gym
    import time

    def run():
        env = gym.make('gym_anm:ANM6Easy-v0')
        o = env.reset()

        for i in range(100):
            a = env.action_space.sample()
            o, r, done, info = env.step(a)
            env.render()
            time.sleep(0.5)  # otherwise the rendering is too fast for the human eye.

        env.close()

    if __name__ == '__main__':
        run()

For more information about the Gym interface, read the `official documentation <https://github.com/openai/gym>`_.


Designing your own ANM task
--------------------------------
The :code:`gym-anm` framework is designed to allow users to easily design their own ANM tasks. New
tasks can be implemented by creating a sub-class of :code:`ANMEnv`. The code below provides the
general template to follow: ::

    from gym_anm.envs import ANMEnv

    class MyANMEnv(ANMEnv):

        def __init__(self):
            network = {'baseMVA':..., 'bus':..., 'device':..., 'branch':...}
            observation = [('bus_p', [0,1], 'MW'), ('dev_q', [2], 'MW')]  # or a callable
            K = 1
            delta = 0.25
            gamma = 0.999
            lamb = 1000     # 'lambda' is a reserved keyword in Python
            seed = None
            super(MyANMEnv, self).__init__(network, observation, K, delta, gamma, lamb, seed)

        def init_state(self):
            ...

        def next_vars(self, s_t):
            ...

        def observation_bounds(self):  # optional
            ...

For more information about designing your own :code:`gym-anm` environments, see :ref:`design_new_env`.
