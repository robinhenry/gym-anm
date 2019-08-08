from multiprocessing import Process
from http.server import HTTPServer, SimpleHTTPRequestHandler
from websocket_server import WebsocketServer
from websocket import create_connection
import webbrowser
import json


class WsServer(object):

    def __init__(self):
        self.PORT = 9001
        self.HOST = '127.0.0.1'
        self.address = 'ws://' + self.HOST + ':' + str(self.PORT) + '/'
        self.clients = {}

        self.process = self._start_process()

    def _start_process(self):
        """ Start a WebSocket server as a separate process. """
        process = Process(target=self._start_server)
        process.start()
        return process


    def _start_server(self):
        """ Start a WebSocket server. """
        server = WebsocketServer(self.PORT)
        server.set_fn_client_left(self.client_left)
        server.set_fn_new_client(self.new_client)
        server.set_fn_message_received(self.msg_received)
        server.run_forever()

    def client_left(self, client, server):
        """ Remove client from the list. """
        try:
            self.clients.pop(client['id'])
        except:
            print("Error in removing client %s" % client['id'])

    def new_client(self, client, server):
        """ Add new client to the list. """
        id = client['id']
        self.clients[id] = client

    def msg_received(self, client, server, msg):
        """ Forward received message to all clients. """
        server.send_message_to_all(msg)


class HttpServer(object):
    def __init__(self):
        self.PORT = 8000
        self.HOST = '127.0.0.1'
        self.address = self.HOST + ':' + str(self.PORT)

        self.process = self._start_http_process()

    def _start_http_process(self):
        """ Start an HTTP server as a different process. """
        service = Process(name='http_server', target=self._start_http_server)
        service.start()
        return service

    def _start_http_server(self):
        """ Start an HTTP server. """
        httpd = HTTPServer((self.HOST, self.PORT), SimpleHTTPRequestHandler)
        print('Serving HTTP at : ' + self.HOST + ':' + str(self.PORT) + '...')
        httpd.serve_forever()

def start():
    """ Initialize servers and client (browser window). """
    http_server = HttpServer()
    ws_server = WsServer()
    webbrowser.open_new(http_server.address)

    return http_server, ws_server

def update(ws_address, new_value):
    """ Send an update message through WebSocket with a new value. """
    ws = create_connection(ws_address)
    message = json.dumps({'new_value': new_value})
    ws.send(message)
    ws.close()

def close(http_server, ws_server):
    """ Terminate processes running in parallels. """
    http_server.process.terminate()
    ws_server.process.terminate()


if __name__ == '__main__':
    http_server, ws_server = start()

    while True:
        for i in range(100):
            update(ws_server.address, i)
        for i in range(100):
            update(ws_server.address, 100 - i)
