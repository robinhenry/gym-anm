import ast
import datetime as dt
import os
import time

import pandas as pd

from gym_anm.constants import RENDERED_NETWORK_SPECS, RENDERED_STATE_VALUES
from gym_anm.envs import ANMEnv
from gym_anm.envs.utils import write_html
from gym_anm.envs.smartgrid_env6.rendering.py import rendering
from gym_anm import ROOT_FOLDER
from .network import network


class ANM6(ANMEnv):

    metadata = {'render.modes': ['human', 'save']}

    def __init__(self):

        # Initialize environment.
        seed = None
        obs_values = ['P_DEV', 'Q_DEV', 'SOC']
        delta_t = 15
        super().__init__(network, obs_values, delta_t, seed)

        ### RENDERING ###
        env_folder = os.path.dirname(os.path.realpath(__file__))
        rendering_folder = os.path.join(env_folder, 'rendering')
        rel_path = os.path.join()


        self.svg_data = {'network': os.path.pardir()}

        RENDERING_FOLDER = os.path.join(ROOT_FOLDER, 'rendering')
        RENDERING_LOGS = os.path.join(RENDERING_FOLDER, 'logs')
        ENV_FILES = {'case': 'network.py',
                     'network': 'network.svg',
                     'svgLabels': 'svgLabels.js'}
        WEB_FILES = {'index': 'index.html'}


        # Folder to new environment (== this folder).


        # Store paths to files needed for rendering.
        rel_path = os.path.relpath(env_folder, RENDERING_FOLDER)
        self.svg_data = {'network': os.path.join(rel_path, ENV_FILES['network'])}
        self.svg_data['labels'] = os.path.join(rel_path, ENV_FILES['svgLabels'])
        #####

    def render(self, mode='human', sleep_time=0.1):
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : {'human', 'save'}, optional
            The mode of rendering. If 'human', the environment is rendered while
            the agent interacts with it. If 'save', the state history is saved
            for later visualization.
        sleep_time : float, optional
            The sleeping time between two visualization updates.

        Raises
        ------
        NotImplementedError
            If a non-valid mode is specified.

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
                raise NotImplementedError

            self.render_mode = mode
            specs = [list(self.network_specs[s]) for s in RENDERED_NETWORK_SPECS]
            self._init_render(specs)

            # Render the initial state.
            self.render(mode=mode, sleep_time=1.)

        else:
            state_values = [list(self.state[s]) for s in RENDERED_STATE_VALUES]
            self._update_render(self.time - self.timestep_length,
                                state_values,
                                list(self.P_gen_potential),
                                [self.e_loss, self.penalty],
                                sleep_time=sleep_time)

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

        if self.render_mode in ['human', 'replay']:
            write_html(self.svg_data)
            self.http_server, self.ws_server = rendering.start(
                title,
                *network_specs)

        elif self.render_mode == 'save':
            s = pd.Series({'title': title, 'specs': network_specs})
            self.render_history = pd.DataFrame([s])

        else:
            raise NotImplementedError

    def _update_render(self, cur_time, state_values, P_potential, costs,
                       sleep_time):
        """
        Update the rendering of the environment state.

        Parameters
        ----------
        cur_time : datetime.datetime
            The time corresponding to the current time step.
        state_values : list of list of float
            The state values needed for rendering.
        P_potential : list of float
            The potential generation of each VRE before curtailment (MW).
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
            rendering.update(self.ws_server.address,
                             cur_time,
                             *state_values,
                             P_potential,
                             costs)
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
            `render_mode` is 'save'.

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