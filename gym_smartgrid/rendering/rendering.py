from multiprocessing import Process
from http.server import HTTPServer, SimpleHTTPRequestHandler
from websocket_server import WebsocketServer
from websocket import create_connection
import webbrowser
import json
import os
import time


class WsServer(object):

    def __init__(self):
        self.PORT = 9001
        self.HOST = '127.0.0.1'
        self.address = 'ws://' + self.HOST + ':' + str(self.PORT) + '/'
        self.clients = {}
        self.init_message = None
        self.init_client = None

        self.process = self._start_process()

    def _start_process(self):
        process = Process(target=self._start_server)
        process.start()
        return process


    def _start_server(self):
        server = WebsocketServer(self.PORT)
        server.set_fn_client_left(self.client_left)
        server.set_fn_new_client(self.new_client)
        server.set_fn_message_received(self.msg_received)
        server.run_forever()

    def client_left(self, client, server):
        try:
            self.clients.pop(client['id'])
        except:
            print("Error in removing client %s" % client['id'])

    def new_client(self, client, server):
        id = client['id']
        self.clients[id] = client

        if self.init_client is None:
            self.init_client = client

        if id != self.init_client['id']:
            server.send_message(client, self.init_message)

    def msg_received(self, client, server, msg):
        message = json.loads(msg)
        if message['messageLabel'] == 'init':
            self.init_message = msg
        elif message['messageLabel'] == 'update':
            server.send_message_to_all(msg)

class HttpServer(object):
    def __init__(self):
        self.PORT = 8000
        self.HOST = '127.0.0.1'
        self.address = self.HOST + ':' + str(self.PORT)
        self.process = self._start_http_process()

    def _start_http_process(self):
        service = Process(name='http_server', target=self._start_http_server)
        service.start()
        return service

    def _start_http_server(self):
        web_dir = os.path.dirname(__file__)
        os.chdir(web_dir)

        httpd = HTTPServer((self.HOST, self.PORT), SimpleHTTPRequestHandler)
        print('Serving HTTP at : ' + self.HOST + ':' + str(self.PORT) + '...')
        httpd.serve_forever()

def start(dev_type, p_min, p_max, i_max, soc_min, soc_max):
    http_server = HttpServer()
    ws_server = WsServer()
    webbrowser.open_new(http_server.address)

    ws = create_connection(ws_server.address)

    message = json.dumps({'messageLabel': 'init',
                          'deviceType': dev_type,
                          'pMin': p_min,
                          'pMax': p_max,
                          'iMax': i_max,
                          'socMin': soc_min,
                          'socMax': soc_max},
                         separators=(',', ':'))
    ws.send(message)
    ws.close()

    return http_server, ws_server

def update(ws_address, cur_time, p, i, soc, p_branch, p_potential):
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
    http_server.process.terminate()
    ws_server.process.terminate()
