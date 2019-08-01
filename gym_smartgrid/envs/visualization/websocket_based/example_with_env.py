from gym_smartgrid.envs.smartgrid_env2 import SmartGridEnv2
from gym_smartgrid.envs.visualization.websocket_based.server import start_http_server

from websocket import create_connection
from multiprocessing import Process
from websocket_server import WebsocketServer
import time
import webbrowser
import numpy as np

clients = {}

class MyWsServer(object):

    def __init__(self, env):
        server_process = Process(target=self._start_server)
        self.process = server_process.start()

        self.animation_cliend_id = '1'


    def _start_server(self):
        server = WebsocketServer(9001)
        server.set_fn_client_left(self.client_left)
        server.set_fn_new_client(self.new_client)
        server.set_fn_message_received(self.msg_received)
        server.run_forever()


    def client_left(self, client, server):
        msg = "Client (%s) left" % client['id']
        print(msg)
        try:
            clients.pop(client['id'])
        except:
            print("Error in removing client %s" % client['id'])
        for cl in clients.values():
            #server.send_message(cl, msg)
            pass


    def new_client(self, client, server):
        msg = "New client (%s) connected" % client['id']
        print(msg)
        for cl in clients.values():
            #server.send_message(cl, msg)
            pass
        clients[client['id']] = client


    def msg_received(self, client, server, msg):
        #msg = "Client (%s) : %s" % (client['id'], msg)
        print("Client (%s) : %s" % (client['id'], msg))

        server.send_message_to_all((str(msg)))



def _update_visualization(obs):
    ws = create_connection('ws://127.0.0.1:9001')
    ws.send(str(obs))
    ws.close()


if __name__ == '__main__':

    # Start http and websocket servers in different processes.
    http_service = start_http_server()
    ws_server = MyWsServer(2)
    webbrowser.open_new('0.0.0.0:8000')
    time.sleep(5)


    # Create clients and connect to the websocket server.
    a = 50
    for i in range(10000):
        a += np.random.randint(-1, 2)

        _update_visualization(a)
