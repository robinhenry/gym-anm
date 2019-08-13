import webbrowser
import json

from websocket import create_connection

from gym_smartgrid.rendering.py.servers import WsServer, HttpServer


def start(dev_type, p_min, p_max, i_max, soc_max):
    """
    Start visualizing the state of the environment in a new browser window.

    Parameters
    ----------
    dev_type : list of int
        The type of each device connected to the network.
    p_min, p_max : list of float
        The minimum and maximum real power injection of each device (MW).
    i_max : list of float
        The transmission line current ratings (p.u.).
    soc_max : list of float
        The maximum state of charge of each storage unit (MWh).

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
    webbrowser.open_new(http_server.address + '/rendering')

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
                          'iMax': i_max,
                          'socMin': [0.] * len(soc_max),
                          'socMax': soc_max},
                         separators=(',', ':'))
    ws.send(message)
    ws.close()

    return http_server, ws_server

def update(ws_address, cur_time, p, i, soc, p_branch, p_potential):
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
    i : list of float
        The current magnitude flow in each transmission line (p.u.).
    soc : list of float
        The state of charge of each storage unit (MWh).
    p_branch : list of float
        The real power flow in each transmisssion line (MW).
    p_potential : list of float
        The potential real power generation of each VRE device before curtailment
        (MW).
    """

    ws = create_connection(ws_address)

    time_array = [cur_time.month, cur_time.day, cur_time.hour, cur_time.minute]
    message = json.dumps({'messageLabel': 'update',
                          'time': time_array,
                          'pInjections': p,
                          'iCurrents': i,
                          'socStorage': soc,
                          'pBranchFlows': p_branch,
                          'pPotential': p_potential})
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
