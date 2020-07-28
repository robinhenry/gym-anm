import ast
import datetime as dt
import time

import pandas as pd

from gym_anm.envs import ANMEnv
from gym_anm.envs.anm6_env.rendering.py import rendering
from gym_anm.envs.anm6_env.network import network


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
                 aux_bounds=None, seed=None):

        super().__init__(network, observation, K, delta_t, gamma, lamb,
                         aux_bounds, seed)

        # Time variables.
        # self.timestep_length = dt.timedelta(minutes=int(60 * delta_t))
        # self.year = 2020

        # Specs of the network (used for rendering).
        self.network_specs = self.simulator.get_rendering_specs()

        # # Rendered values.
        # self.rendered_network_specs = ['dev_type', 'dev_p', 'branch_s', 'des_soc']
        # self.rendered_state_values = ['dev_p', 'branch_s', 'des_soc', 'gen_p_max']

    def render(self, mode='human', sleep_time=0.):
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : {'human', 'save'}, optional
            The mode of rendering. If 'human', the environment is rendered while
            the agent interacts with it. If 'save', the state history is saved
            for later visualization using the function replay().
        sleep_time : float, optional
            The sleeping time between two visualization updates, only used if
            mode == 'human'. Default value is 0.

        Raises
        ------
        NotImplementedError
            If a non-valid mode is specified.

        Notes
        -----
        When using mode == 'save', do not forget to call close() to stop
        saving the history of interactions. If no call to close() is made,
        the history will not be saved.

        Using mode == 'human' with `sleep_time` > 0 will pause the environment
        during `sleep_time` seconds, which will significantly slow down agent-environment
        interactions. It is therefore recommended not to render the environment during
        training.

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
            rendered_network_specs = ['dev_type', 'dev_p', 'branch_s', 'des_soc']
            specs = {s : self.network_specs[s] for s in rendered_network_specs}
            self._init_render(specs)

            # Render the initial state.
            self.render(mode=mode, sleep_time=1.)

        else:
            full_state = self.simulator.state
            dev_p = list(full_state['dev_p']['MW'].values())
            branch_s = list(full_state['branch_s']['MVA'].values())
            des_soc = list(full_state['des_soc']['MWh'].values())
            gen_p_max = list(full_state['gen_p_max']['MW'].values())
            costs = [self.e_loss, self.penalty]
            cur_time = dt.datetime(2020, 1, 1, 1)

            self._update_render(cur_time, dev_p, branch_s, des_soc, gen_p_max,
                                costs, sleep_time)

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
        dev_type = list(network_specs['dev_type'])
        p_min, p_max = [], []
        for i in network_specs['dev_p'].keys():
            p_min.append(network_specs['dev_p'][i]['MW'][0])
            p_max.append(network_specs['dev_p'][i]['MW'][1])
        branch_rate = []
        for br in network_specs['branch_s'].keys():
            branch_rate.append(network_specs['branch_s'][br]['pu'][1])
        soc_min, soc_max = [], []
        for i in network_specs['des_soc'].keys():
            soc_min.append(network_specs['des_soc'][i]['MWh'][0])
            soc_max.append(network_specs['des_soc'][i]['MWh'][1])

        # Add the '-' to the displayed title.
        if 'Easy' in title:
            title = 'ANM6-Easy'

        if self.render_mode in ['human', 'replay']:
            rendering.write_html()
            self.http_server, self.ws_server = \
                rendering.start(title, dev_type, p_min, p_max, branch_rate, soc_min,
                                soc_max)

        elif self.render_mode == 'save':
            s = pd.Series({'title': title, 'specs': network_specs})
            self.render_history = pd.DataFrame([s])

        else:
            raise NotImplementedError

    def _update_render(self, cur_time, dev_p, branch_s, des_soc, gen_p_max,
                       costs, sleep_time):
        """
        Update the rendering of the environment state.

        Parameters
        ----------
        cur_time : datetime.datetime
            The time corresponding to the current time step.
        dev_p  : list of float
            The real power injection from each device (MW).
        branch_s : list of float
            The apparent power flow in each branch (MVA).
        des_soc : list of float
            The state of charge of each storage unit (MWh).
        gen_p_max : list of float
            The potential real power generation of each RER generator before
            curtailment (MW).
        costs : list of float
            The total energy loss and the total penalty associated with operating
            constraints violation.
        sleep_time : float
            The sleeping time between two visualization updates.

        Raises
        ------
        NotImplementedError
            If the rendering mode is non-valid.
        """

        if self.render_mode in ['human', 'replay']:
            rendering.update(self.ws_server.address, cur_time, dev_p, branch_s,
                             des_soc, gen_p_max, costs)
            time.sleep(sleep_time)

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
