from gym_smartgrid.envs.visualization.websocket_based.server import start_http_server

from websocket import create_connection
import time
import webbrowser
import json

from gym_smartgrid.visualization.rendering import WsServer




def init_visualization():
    # Send the initial structure of the graph.
    ws = create_connection('ws://127.0.0.1:9001')
    p_min = [0, 0, 0, 10, 0, 0, 20]
    p_max = [100, 200, 500, 50, 20, 200, 70]
    i_max = {'(0, 1)': 100, '(1, 2)': 200, '(1, 3)': 100, '(2, 4)': 50, '(2, 5)': 300}
    soc_min = [5]
    soc_max = [200]
    message = json.dumps({'messageLabel': 'init', 'pMin': p_min, 'pMax': p_max,
                          'iMax': i_max, 'socMin': soc_min, 'socMax': soc_max})
    ws.send((message))
    ws.close()

def update_visualization(obs):
    ws = create_connection('ws://127.0.0.1:9001')
    i = [50] * 5
    soc = [100]
    message = json.dumps({'messageLabel': 'update', 'pInjections': obs,
                          'iCurrents': i, 'socStorage': soc})
    ws.send(message)
    ws.close()


if __name__ == '__main__':

    # Start http and websocket servers in different processes.
    http_service = start_http_server()
    ws_server = WsServer(2)
    webbrowser.open_new('0.0.0.0:8000')
    time.sleep(2)

    init_visualization()

    # Create clients and connect to the websocket server.
    a = 0
    while True:
        for i in range(200):
            a += 1
            update_visualization([100] * 7)
        for i in range(200):
            a -= 1
            update_visualization([100] * 7)
