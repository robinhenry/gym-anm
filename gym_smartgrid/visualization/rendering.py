from multiprocessing import Process
from http.server import HTTPServer, SimpleHTTPRequestHandler
from websocket_server import WebsocketServer
from websocket import create_connection
import webbrowser
import json


class WsServer(object):

    def __init__(self):
        server_process = Process(target=self._start_server)
        self.PORT = 9001
        self.HOST = '127.0.0.1'
        self.process = server_process.start()
        self.address = 'ws://' + self.HOST + ':' + str(self.PORT)
        self.clients = {}

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
        self.clients[client['id']] = client

    def msg_received(self, client, server, msg):
        server.send_message_to_all((str(msg)))

class HttpServer(object):
    def __init__(self):
        self.PORT = 8000
        self.HOST = '0.0.0.0'
        self.address = self.HOST + ':' + str(self.HOST)
        self.process = self._start_http_process()

    def _start_http_process(self):
        service = Process(name='http_server', target=self._start_http_server)
        service.start()
        return service

    def _start_http_server(self):
        httpd = HTTPServer((self.HOST, self.PORT), SimpleHTTPRequestHandler)
        print('Serving HTTP at : ' + self.HOST + ':' + str(self.PORT) + '...')
        httpd.serve_forever()

def start(p_min, p_max, i_max, soc_min, soc_max):
    http_server = HttpServer()
    ws_server = WsServer()
    webbrowser.open(http_server.address, new=1)

    ws = create_connection(ws_server.address)
    message = json.dumps(
        {'messageLabel': 'init', 'pMin': p_min, 'pMax': p_max,
         'iMax': i_max, 'socMin': soc_min, 'socMax': soc_max})
    ws.send((message))
    ws.close()

    return http_server, ws_server

def update(ws_address, p, i, soc):
    ws = create_connection(ws_address)
    message = json.dumps({'messageLabel': 'update', 'pInjections': p,
                          'iCurrents': i, 'socStorage': soc})
    ws.send(message)
    ws.close()

def close(http_server, ws_server):
    http_server.process.terminate()
    ws_server.process.terminate()
