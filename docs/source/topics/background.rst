..

Background and notation
=======================

Reinforcement learning
----------------------
The documentation of :code:`gym-anm` assumes familiarity with basic reinforcement learning (RL) concepts. Some good
resources to get started are:

* `Reinforcement Learning: An Introduction <http://incompleteideas.net/book/the-book.html>`_
* `OpenAI Spinning Up <https://spinningup.openai.com/en/latest/index.html>`_

Being familiar with the `OpenAI Gym <https://gym.openai.com/>`_ framework is also useful, since :code:`gym-anm`
environments follow the same framework.


Distribution networks
---------------------

Notation
^^^^^^^^
The main notations used throughout this documentation are listed below.

* :math:`\mathbf i` - the imaginary number with :math:`\mathbf i^2 = -1`.
* :math:`G(\mathcal N, \mathcal E)` - the directed graph representing the distribution network,
* :math:`\mathcal N = \{0,1,\ldots,N-1\}` - the set of buses (or nodes) in the network,
* :math:`\mathcal E \subseteq \mathcal N \times \mathcal N` - the set of directed edges (transmission lines) linking buses together,
* :math:`e_{ij} \in \mathcal E` - the directed edge with sending bus :math:`i` and receiving bus :math:`j`,
* :math:`\mathcal D = \{0,1,\ldots,D-1\}` - the set of all electrical devices connected to the grid. Each device
  :math:`d \in \mathcal D` is connected to exactly one bus and may inject or withdraw power into/from the grid.
* :math:`\mathcal D_i \subseteq \mathcal D` - the set of electrical devices connected to bus :math:`i`,
* :math:`V_i, I_i, P_i^{(bus)}, Q_i^{(bus)}` - the complex voltage level, complex total current injection, total real power injection,
  and total reactive power injection at bus :math:`i`, respectively,
* :math:`P_d^{(dev)}, Q_d^{(dev)}` - the real and reactive power injections of device :math:`d \in \mathcal D` into the grid,
  respectively,
* :math:`I_{ij}, P_{ij}, Q_{ij}, S_{ij}` - the complex current, active power flow, reactive power flow, and complex
  power flow in branch :math:`e_{ij} \in \mathcal E`, from bus :math:`i` to bus :math:`j`, respectively, with
  :math:`S_{ij} = P_{ij} + \mathbf i Q_{ij}`.
* :math:`\mathcal D_L \subset \mathcal D` - the set of passive load devices that only withdraw power from the grid,
* :math:`\mathcal D_G \subset \mathcal D` - the set of generators, which only inject power into the grid, with the
  exception the slack device (see below),
* :math:`\mathcal D_{DES} \subset \mathcal D` - the set of distributed energy storage (DES) units, which can both
  inject and withdraw power into/from the grid,
* :math:`\mathcal D_{DER} \subset \mathcal D_G` - the set of renewable energy resources (a subset of all generators),
* :math:`g^{slack} \in \mathcal D_G - \mathcal D_{DER}` - the slack device, a generator used to balance power flow in
  the network and provide a voltage reference. The slack device is the only device connected to the slack bus :math:`i=0`,
* :math:`SoC_d` - the state of charge (i.e., energy level) of storage unit :math:`d \in \mathcal D_{DES}`,
* :math:`P_g^{(max)}` - the maximum real power that generator :math:`g \in \mathcal D_G - \{g^{slack}\}` can produce if
  not curtailed,

Basic concepts and assumptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The slack bus is assumed unique and at :math:`i=0`, with a fixed voltage reference :math:`V_0 = 1 \angle 0`.

Unless otherwise stated, all electrical quantities are expressed in `per unit (p.u.) <https://en.wikipedia.org/wiki/Per-unit_system>`_.

The power grid is assumed to be a `three-phase balanced system <https://en.wikipedia.org/wiki/Three-phase>`_ and we
adopt its single-phase equivalent representation in all derivations.

For a more in-depth description of the power grid model used in :code:`gym-anm`, see the `original paper ADD LINK`_.
