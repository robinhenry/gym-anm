..

Rendering an Environment
========================
It is often desirable to be able to watch your agent interacting with the environment (and it makes the whole
process more fun!). Currently, :code:`gym-anm` does not, however, support the rendering of arbitrary environments.

The only exception is the initial task :code:`ANM6Easy-v0`, for which a web-based rendering tool is available
(through the :code:`env.render()` and :code:`env.close()` calls).

In addition, the implementation was designed so as to make it easy for others to build variants of
:code:`ANM6Easy-v0`, while benefiting from its rendering tool. This can be achieved by creating your environment
as a sub-class of :py:class:`gym_anm.envs.anm6_env.anm6.ANM6`. By doing so, you will automatically inherit the
same 6-bus distribution power grid (defined by `this network dictionary
<https://github.com/robinhenry/gym-anm/blob/master/gym_anm/envs/anm6_env/network.py>`_).

After slightly modifying the example from the previous page, the code below shows a custom environment
that inherits the 6-bus power grid used in :code:`ANM6Easy-v0` and therefore makes its rendering possible
to its users.

.. literalinclude:: ../../../examples/custom_anm6.py
   :language: python
