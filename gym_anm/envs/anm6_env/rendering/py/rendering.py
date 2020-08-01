import webbrowser
import json
import os

from websocket import create_connection
from .servers import WsServer, HttpServer
from .constants import RENDERING_FOLDER, RENDERING_RELATIVE_PATH


def start(title, dev_type, p_max, q_max, s_rate, v_magn_min, v_magn_max, soc_max,
          costs_range):
    """
    Start visualizing the state of the environment in a new browser window.

    Parameters
    ----------
    title : str
        The title to give to the visualization, usually the name of the
        environment.
    dev_type : list of int
        The type of each device connected to the network.
    p_max : list of float
        The maximum absolute real power injection of each device (MW).
    q_max : list of float
        The maximum absolute reactive power injection of each device (MVAr).
    s_rate : list of float
        The transmission line apparent power ratings (MVA).
    v_magn_min : list of float
        The minimum voltage magnitude allowed at each bus (pu).
    v_magn_max : list of float
        The maximum voltage magnitude allowed at each bus (pu).
    soc_max : list of float
        The maximum state of charge of each storage unit (MWh).
    costs_range : tuple of int
        The maximum absolute energy loss costs_clipping[0] and the maximum
        constraints violation penalty (parts of the reward function).

    Returns
    -------
    http_server : HttpServer
        The HTTP server serving the visualization.
    ws_server : WsServer
        The WebSocket server used for message exchanges between the environment
        and the visualization.
    """

    # Initialize the servers.
    http_server = HttpServer()
    ws_server = WsServer()

    # Open a new browser window to display the visualization.
    p = os.path.join(http_server.address, RENDERING_RELATIVE_PATH)
    webbrowser.open_new_tab(p)

    ws = create_connection(ws_server.address)

    message = json.dumps({'messageLabel': 'init',
                          'deviceType': dev_type,
                          'pMax': p_max,
                          'qMax': q_max,
                          'sRate': s_rate,
                          'vMagnMin': v_magn_min,
                          'vMagnMax': v_magn_max,
                          'socMax': soc_max,
                          'energyLossMax': costs_range[0],
                          'penaltyMax': costs_range[1],
                          'title': title},
                         separators=(',', ':'))
    ws.send(message)
    ws.close()

    return http_server, ws_server


def update(ws_address, cur_time, year_count, p, q, s, soc, p_potential,
           bus_v_magn, costs, network_collapsed):
    """
    Update the visualization of the environment.

    Parameters
    ----------
    ws_address : str
        The address of the listening WebSocket server.
    cur_time : datetime.datetime
        The time corresponding to the state of the network.
    year_count : int
        The number of full years passed since the last reset of the environment.
    p  : list of float
        The real power injection from each device (MW).
    q : list of float
        The reactive power injection from each device (MVAr).
    s : list of float
        The apparent power flow in each branch (MVA).
    soc : list of float
        The state of charge of each storage unit (MWh).
    p_potential : list of float
        The potential real power generation of each VRE device before curtailment
        (MW).
    bus_v_magn : list of float
        The voltage magnitude of each bus (pu).
    costs : list of float
        The total energy loss and the total penalty associated with operating
        constraints violation.
    network_collapsed : bool
        True if no load flow solution is found (possibly infeasible); False
        otherwise.
    """

    ws = create_connection(ws_address)

    time_array = [cur_time.month, cur_time.day, cur_time.hour, cur_time.minute]
    message = json.dumps({'messageLabel': 'update',
                          'time': time_array,
                          'yearCount': year_count,
                          'pInjections': p,
                          'qInjections': q,
                          'sFlows': s,
                          'socStorage': soc,
                          'pPotential': p_potential,
                          'vMagn' : bus_v_magn,
                          'reward': costs,
                          'networkCollapsed': network_collapsed})
    ws.send(message)
    ws.close()

    return


def close(http_server, ws_server):
    """
    Terminate the parallel processes running the HTTP and WebSocket servers.

    Parameters
    ----------
    http_server : HttpServer
        The HTTP server serving the visualization.
    ws_server : WsServer
        The WebSocket server used for message exchanges between the environment
        and the visualization.
    """

    http_server.process.terminate()
    ws_server.process.terminate()


def write_html():
    """
    Update the index.html file used for rendering the environment state.
    """

    s = """<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="css/styles.css">
    <script src="js/init.js"></script>
    <script src="js/devices.js"></script>
    <script src="js/graph.js"></script>
    <script src="js/dateTime.js"></script>
    <script src="js/reward.js"></script>
    <script src="js/text.js"></script>
    <script src="envs/anm6/svgLabels.js"></script>
    <title>gym-anm:ANM6</title>
</head>

<body onload="init();">

    <header></header>

    <object id="svg-network" data="envs/anm6/network_2.svg"
            type="image/svg+xml" class="network">
    </object>

</body>
</html>

    """

    html_file = os.path.join(RENDERING_FOLDER, 'index.html')

    with open(html_file, 'w') as f:
        f.write(s)
