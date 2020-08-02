import ast
import datetime as dt
import time
import numpy as np
import pandas as pd

from ..anm_env import ANMEnv
from .rendering.py import rendering
from .network import network
from .utils import random_date


class ANM6(ANMEnv):
    """
    The base class for a 6-bus and 7-device `gym-anm` environment.

    The structure of the electricity distribution network used for this
    environment is shown below:

    Slack ----------------------------
            |            |           |
          -----       -------      -----
         |     |     |       |    |     |
        House  PV  Factory  Wind  EV   DES

    This environment supports rendering (web-based) through the functions
    render(), replay(), and close().
    """

    metadata = {'render.modes': ['human', 'save']}

    def __init__(self, observation, K, delta_t, gamma, lamb,
                 aux_bounds=None, costs_clipping=(None, None), seed=None):

        super().__init__(network, observation, K, delta_t, gamma, lamb,
                         aux_bounds, costs_clipping, seed)

        # Rendering variables.
        self.network_specs = self.simulator.get_rendering_specs()
        self.timestep_length = dt.timedelta(minutes=int(60 * delta_t))
        self.date = None
        self.date_init = None
        self.year_count = 0
        self.skipped_frames = None

    def render(self, mode='human', skip_frames=0):
        """
        Render the current state of the environment.

        Visualizing the agent-environment interactions in real-time (e.g.,
        during training) is hard to follow and not very useful, as the state of
        the distribution network changes too quickly (you can try with
        `mode`='human' and `skip_frames=0`). Instead, setting `skip_frames`>0
        will only update the rendering of the environment every `skip_frames`+1
        steps (assuming `render(skip_frames)` is called after every step),
        which will make it much easier to follow for the human eye.

        Parameters
        ----------
        mode : {'human', 'save'}, optional
            The mode of rendering. If 'human', the environment is rendered while
            the agent interacts with it. If 'save', the state history is saved
            for later visualization using the function replay().
        skip_frames : int, optional
            The number of frames (steps) to skip when rendering the environment.
            For example, `skip_frames`=3 will update the rendering of the
            environment every 4 calls to `render()`.

        Raises
        ------
        NotImplementedError
            If a non-valid mode is specified.

        Notes
        -----
        1. The use of `mode`='human' and `skip_frames`>0 assumes that `render()`
        is called after each step the agent takes in the environment.
        The same behavior can be achieved with `skip_frames`=0 and calling
        `render()` less frequently.
        2. When using mode == 'save', do not forget to call close() to stop
        saving the history of interactions. If no call to close() is made,
        the history will not be saved.

        See Also
        --------
        replay()
        """

        if self.render_mode is None:
            if mode in ['human', 'replay']:
                pass
            elif mode == 'save':
                self.render_history = None
            else:
                raise NotImplementedError()

            # Render the initial image of the distribution network.
            self.render_mode = mode
            rendered_network_specs = ['dev_type', 'dev_p', 'dev_q', 'branch_s',
                                      'bus_v', 'des_soc']
            specs = {s : self.network_specs[s] for s in rendered_network_specs}
            self._init_render(specs)

            # Render the initial state.
            # NOTE: a sleep time of 2sec is hard-coded here, to make sure that
            # the rendering initialization step is finished before moving on.
            # time.sleep(2.)
            self.render(mode=mode, skip_frames=skip_frames)

        else:
            self.skipped_frames = (self.skipped_frames + 1) % (skip_frames + 1)
            if self.skipped_frames:
                return

            full_state = self.simulator.state
            dev_p = list(full_state['dev_p']['MW'].values())
            dev_q = list(full_state['dev_q']['MVAr'].values())
            branch_s = list(full_state['branch_s']['MVA'].values())
            des_soc = list(full_state['des_soc']['MWh'].values())
            gen_p_max = list(full_state['gen_p_max']['MW'].values())
            bus_v_magn = list(full_state['bus_v_magn']['pu'].values())
            costs = [self.e_loss, self.penalty]
            network_collapsed = not self.simulator.pfe_converged

            self._update_render(dev_p, dev_q, branch_s, des_soc,
                                gen_p_max, bus_v_magn, costs, network_collapsed)

    def step(self, action):
        obs, r, done, info = super().step(action)

        # Increment the date (for rendering).
        self.date += self.timestep_length

        # Increment the year count.
        self.year_count = (self.date - self.date_init).days // 365

        return obs, r, done, info

    def reset(self, date_init=None):
        obs = super().reset()
        self.skipped_frames = 0

        # Reset the date (for rendering).
        self.year_count = 0
        if date_init is None:
            self.date_init = random_date(self.np_random, 2020)
        else:
            self.date_init = date_init
        self.date = self.date_init

        return obs

    def _reset_date(self, date_init):
        """Reset the date displayed in the visualization (and the year count)."""
        self.date_init = date_init
        self.date = date_init

    def _init_render(self, network_specs):
        """
        Initialize the rendering of the environment state.

        Parameters
        ----------
        network_specs : dict of {str : list}
            The operating characteristics of the electricity distribution network.

        Raises
        ------
        NotImplementedError
            If the rendering mode is non-valid.
        """

        title = type(self).__name__

        # Convert dict of network specs into lists.
        dev_type = list(network_specs['dev_type'].values())
        ps = []
        qs = []
        for i in network_specs['dev_p'].keys():
            p_min_max = [network_specs['dev_p'][i]['MW'][j] for j in [0, 1]]
            ps.append(np.max(np.abs(p_min_max)))
            q_min_max = [network_specs['dev_q'][i]['MVAr'][j] for j in [0, 1]]
            qs.append(np.max(np.abs(q_min_max)))
        branch_rate = []
        for br in network_specs['branch_s'].keys():
            branch_rate.append(network_specs['branch_s'][br]['MVA'][1])
        bus_v_min, bus_v_max = [], []
        for i in network_specs['bus_v'].keys():
            bus_v_min.append(network_specs['bus_v'][i]['pu'][0])
            bus_v_max.append(network_specs['bus_v'][i]['pu'][1])
        soc_max = []
        for i in network_specs['des_soc'].keys():
            soc_max.append(network_specs['des_soc'][i]['MWh'][1])

        # Add the '-' to the displayed title.
        if 'Easy' in title:
            title = 'ANM6-Easy'

        # Set default costs range if not specified.
        c1 = 100 if self.costs_clipping[0] is None else self.costs_clipping[0]
        c2 = 10000 if self.costs_clipping[1] is None else self.costs_clipping[1]
        costs_range = (c1, c2)

        if self.render_mode in ['human', 'replay']:
            rendering.write_html()
            self.http_server, self.ws_server = \
                rendering.start(title, dev_type, ps, qs, branch_rate,
                                bus_v_min, bus_v_max, soc_max, costs_range)

        elif self.render_mode == 'save':
            s = pd.Series({'title': title, 'specs': network_specs})
            self.render_history = pd.DataFrame([s])

        else:
            raise NotImplementedError

    def _update_render(self, dev_p, dev_q, branch_s, des_soc, gen_p_max,
                       bus_v_magn, costs, network_collapsed):
        """
        Update the rendering of the environment state.

        Parameters
        ----------
        dev_p  : list of float
            The real power injection from each device (MW).
        dev_q : list of float
            The reactive power injection from each device (MW).
        branch_s : list of float
            The apparent power flow in each branch (MVA).
        des_soc : list of float
            The state of charge of each storage unit (MWh).
        gen_p_max : list of float
            The potential real power generation of each RER generator before
            curtailment (MW).
        bus_v_magn : list of float
            The voltage magnitude of each bus (pu).
        costs : list of float
            The total energy loss and the total penalty associated with operating
            constraints violation.
        network_collapsed : bool
            True if no load flow solution is found (possibly infeasible); False
            otherwise.

        Raises
        ------
        NotImplementedError
            If the rendering mode is non-valid.
        """

        if self.render_mode in ['human', 'replay']:
            rendering.update(self.ws_server.address, self.date, self.year_count,
                             dev_p, dev_q, branch_s, des_soc, gen_p_max,
                             bus_v_magn, costs, network_collapsed)

        elif self.render_mode == 'save':
            d = {'time': self.time,
                 'state_values': state_values,
                 'potential': P_potential,
                 'costs': costs}
            s = pd.Series(d)
            self.render_history = self.render_history.append(s,
                                                             ignore_index=True)

        else:
            raise NotImplementedError

    def replay(self, path, sleep_time=0.1):
        """
        Render a state history previously saved.

        Parameters
        ----------
        path : str
            The path to the saved history.
        sleep_time : float, optional
            The sleeping time between two visualization updates.
        """

        self.reset()
        self.render_mode = 'replay'

        history = pd.read_csv(path)
        ns, obs, p_pot, times, costs = self._unpack_history(history)

        self._init_render(ns)

        for i in range(len(obs)):
            self._update_render(times[i], obs[i], p_pot[i], costs, sleep_time)

        self.close()

    def _unpack_history(self, history):
        """
        Unpack a previously stored history of state variables.

        Parameters
        ----------
        history : pandas.DataFrame
            The history of states, with fields {'specs', 'time', 'state_values',
            'potential'}.

        Returns
        -------
        ns : dict of {str : list}
            The operating characteristics of the electricity distribution network.
        state_values : list of list of float
            The state values needed for rendering.
        p_potential : list of float
            The potential generation of each VRE before curtailment (MW).
        times : list of datetime.datetime
            The times corresponding to each time step.
        costs : list of float
            The total energy loss and the total penalty associated with operating
            constraints violation.
        """

        ns = ast.literal_eval(history.specs[0])

        state_values = history.state_values[1:].values
        state_values = [ast.literal_eval(o) for o in state_values]

        p_potential = history.potential[1:].values
        p_potential = [ast.literal_eval(p) for p in p_potential]

        times = history.time[1:].values
        times = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in times]

        costs = history.costs[1:].values
        costs = [ast.literal_eval(c) for c in costs]

        return ns, state_values, p_potential, times, costs

    def close(self, path=None):
        """
        Close the rendering.

        Parameters
        ----------
        path : str, optional
            The path to the file to store the state history, only used if
            `render_mode` == 'save'.

        Returns
        -------
        pandas.DataFrame
            The state history.
        """

        to_return = None

        if self.render_mode in ['human', 'replay']:
            rendering.close(self.http_server, self.ws_server)

        if self.render_mode == 'save':
            if path is None:
                raise ValueError('No path specified to save the history.')
            self.render_history.to_csv(path, index = None, header=True)
            to_return = self.render_history

        self.render_mode = None

        return to_return
