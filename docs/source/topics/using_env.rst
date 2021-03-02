..

Using an Environment
====================

Initializing
-------------
If the :code:`gym-anm` environment you would like to use has already been registered in the :code:`gym`'s registry
(see the `Gym documentation <https://gym.openai.com/docs/#available-environments>`_), you can initialize it with
:code:`gym.make('gym_anm:<ENV_ID>')`, where :code:`<ENV_ID>` it the ID of the environment. For example: ::

    import gym
    env = gym.make('gym_anm:ANM6Easy-v0')

*Note: all environments provided as part of the* :code:`gym-anm` *package are automatically registered.*

Alternatively, the environment can be initialized directly from its class: ::

    from gym_anm.envs import ANM6Easy
    env = ANM6Easy()

Agent-environment interactions
------------------------------
Built on top of `Gym <https://github.com/openai/gym>`_, :code:`gym-anm` provides 2 core functions: :code:`reset()` and
:code:`step(a)`.

:code:`reset()` can be used to reset the environment and collect the first observation of the trajectory: ::

    obs = env.reset()

After the agent has selected an action :code:`a` to apply to the environment, :code:`step(a)` can be used to do so: ::

    obs, r, done, info = env.step(a)

where:

* :code:`obs` is the vector of observations :math:`o_{t+1}`,
* :code:`r` is the reward :math:`r_t`,
* :code:`done` is a boolean value set to :code:`true` if :math:`s_{t+1}` is a terminal state,
* :code:`info` gathers information about the transition (it is seldom used in :code:`gym-anm`).

Render the environment
----------------------
Some :code:`gym-anm` environments may support rendering through the :code:`render()` and :code:`close()` functions.

To update the visualization of the environment, the :code:`render` method is called: ::

    env.render()

To end the visualization and close all used resources: ::

    env.close()

Currently, only :code:`gym-anm:ANM6Easy-v0` supports rendering.

Complete example
----------------
A complete example of agent-environment interactions with an arbitrary agent :code:`agent`: ::

    env = gym.make('gym_anm:ANM6Easy-v0')
    o = env.reset()

    for i in range(1000):
        a = agent.act(o)
        o, r, done, info = env.step(a)
        env.render()
        time.sleep(0.5)   # otherwise the rendering is too fast for the human eye

        if done:
            o = env.reset()

The above example would be rendered in your favorite web browser as:

.. image:: ../images/anm6-easy-example.*
    :width: 800
