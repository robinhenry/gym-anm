..

.. _state_space_label:

State space
===========

Formally, the state vectors used by :code:`gym-anm` are expressed as follows:

.. math::
    \begin{align}
        s_t = \big[
        \{P_{d,t}^{(dev)}\}_{d \in \mathcal D},\; \{Q_{d,t}^{(dev)}\}_{d \in \mathcal D},\; \{SoC_{d,t}\}_{d \in \mathcal D_{DES}},
        \{P_{g,t}^{(max)}\}_{g \in \mathcal D_G - \{g^{slack}\}},\; \{aux^{(k)}_t\}_{k =0}^{K-1} \big] \;, \label{eq:state}
    \end{align}

where:

* :math:`P_{d,t}^{(dev)}` and :math:`Q_{d,t}^{(dev)}` are the real and reactive power injections into the grid from
  electrical device :math:`d \in \mathcal D` at time :math:`t`, respectively,
* :math:`SoC_{d,t}` is the charge level, or state of charge (SoC), of storage unit :math:`d \in \mathcal D_{DES}`,
* :math:`P_{g,t}^{(max)}` is the maximum production that generator :math:`g \in \mathcal D_G - \{g^{slack}\}` can
  produce at time :math:`t`,
* :math:`aux_t^{(k)}` is the value of the :math:`(k-1)^{th}` auxiliary variable generated during the transition from
  timestep :math:`t` to timestep :math:`t+1`.

Terminal states :math:`s \in \mathcal S^{terminal}` are represented by the zero vector :math:`[0,\ldots,0]`.