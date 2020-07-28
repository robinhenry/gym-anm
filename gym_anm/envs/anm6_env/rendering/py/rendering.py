import webbrowser
import json
import os

from websocket import create_connection
from .servers import WsServer, HttpServer
from .constants import RENDERING_FOLDER, RENDERING_RELATIVE_PATH


def start(title, dev_type, p_min, p_max, s_rate, soc_min, soc_max):
    """
    Start visualizing the state of the environment in a new browser window.

    Parameters
    ----------
    title : str
        The title to give to the visualization, usually the name of the
        environment.
    dev_type : list of int
        The type of each device connected to the network.
    p_min, p_max : list of float
        The minimum and maximum real power injection of each device (MW).
    s_rate : list of float
        The transmission line apparent power ratings (MVA).
    soc_min, soc_max : list of float
        The minimum and maximum state of charge of each storage unit (MWh).

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

    # Clip slack device power injection visualization at 0.
    p_min_abs = [0.] + p_min[1:]

    # Clip storage units power injection visualization at 0.
    for i in range(len(p_min_abs)):
        if dev_type[i] == 4:
            p_min_abs[i] = 0.

    message = json.dumps({'messageLabel': 'init',
                          'deviceType': dev_type,
                          'pMin': p_min_abs,
                          'pMax': p_max,
                          'sRate': s_rate,
                          'socMin': [0.] * len(soc_max),
                          'socMax': soc_max,
                          'title': title},
                         separators=(',', ':'))
    ws.send(message)
    ws.close()

    return http_server, ws_server


def update(ws_address, cur_time, p, s, soc, p_potential, costs):
    """
    Update the visualization of the environment.

    Parameters
    ----------
    ws_address : str
        The address of the listening WebSocket server.
    cur_time : datetime.datetime
        The time corresponding to the state of the network.
    p  : list of float
        The real power injection from each device (MW).
    s : list of float
        The apparent power flow in each branch (MVA).
    soc : list of float
        The state of charge of each storage unit (MWh).
    p_potential : list of float
        The potential real power generation of each VRE device before curtailment
        (MW).
    costs : list of float
        The total energy loss and the total penalty associated with operating
        constraints violation.
    """

    ws = create_connection(ws_address)

    time_array = [cur_time.month, cur_time.day, cur_time.hour, cur_time.minute]
    message = json.dumps({'messageLabel': 'update',
                          'time': time_array,
                          'pInjections': p,
                          'sFlows': s,
                          'socStorage': soc,
                          'pPotential': p_potential,
                          'reward': costs})
    ws.send(message)
    ws.close()

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
    <script src="js/title.js"></script>
    <script src="envs/anm6/svgLabels.js"></script>
    <title>SmartGrid-gym</title>
</head>

<body onload="init();">

    <header></header>

    <object id="svg-network" data="envs/anm6/network.svg"
            type="image/svg+xml" class="network">
    </object>

</body>
</html>

    """

    html_file = os.path.join(RENDERING_FOLDER, 'index.html')

    with open(html_file, 'w') as f:
        f.write(s)
