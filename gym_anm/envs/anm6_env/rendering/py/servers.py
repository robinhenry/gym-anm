import json
import os
import sys
import io
from contextlib import redirect_stdout
from http.server import HTTPServer, SimpleHTTPRequestHandler
from multiprocessing import Process
from websocket_server import WebsocketServer
import websocket

from .constants import RENDERING_LOGS, RENDERING_RELATIVE_PATH
from ..... import ROOT_FOLDER


class WsServer(object):
    """
    Encapsulates a WebSocket server.

    Attributes
    ----------
    PORT : int
        The port on which to listen.
    HOST : str
        The host used as server.
    address : str
        The full hosting address to connect to the server.
    clients : dict of {int : dict}
        The clients currently connected to the server.
    init_message : str
        The initial message received, then sent to every new client.
    init_client : dict
        The client which sent the initial message.
    process : multiprocessing.Process
        The separate process running the server.
    """

    def __init__(self):
        self.PORT = 9001
        self.HOST = '127.0.0.1'
        self.address = 'ws://' + self.HOST + ':' + str(self.PORT) + '/'
        self.clients = {}
        self.init_message = None
        self.init_client = None

        # Only start the websocket server if it is not already running.
        try:
            ws = websocket.WebSocket()
            ws.connect(self.address)
            print('Ws connection status: ' + str(ws.connected))
            self.process = None
        except ConnectionRefusedError:
            self.process = self._start_process()

    def _start_process(self):
        """
        Start the server in a parallel process.

        Returns
        -------
        multiprocessing.Process
            The process running the server.
        """

        process = Process(target=self._start_server)
        process.start()

        return process

    def _start_server(self):
        """ Start the WebSocket server and keep it running. """

        # Create websocket server.
        server = WebsocketServer(self.PORT, self.HOST)
        server.set_fn_client_left(self.client_left)
        server.set_fn_new_client(self.new_client)
        server.set_fn_message_received(self.msg_received)

        # Run forever and redirect stdout to .log file.
        f = io.StringIO()
        with redirect_stdout(f):
            server.run_forever()
        with open(os.path.join(os.path.join(RENDERING_LOGS), 'ws_stdout.log'),
                  'w') as file:
            file.write(f.getvalue())

    def new_client(self, client, server):
        """
        Action taken when a new client connects to the server.

        Parameters
        ----------
        client : dict
            The new client.
        server : websocket_server.websocket_server.WebsocketServer
            The current server.
        """

        id = client['id']
        self.clients[id] = client

        if self.init_message is not None and self.init_client is not None:
            if id != self.init_client['id']:
                server.send_message(client, self.init_message)

    def msg_received(self, client, server, msg):
        """
        Action taken when a message is received.

        Parameters
        ----------
        client : dict
            The client that sent the message.
        server : websocket_server.websocket_server.WebsocketServer
            The current server.
        msg : str
            The message received in JSON format.
        """

        message = json.loads(msg)

        if message['messageLabel'] == 'init':
            self.init_client = client
            self.init_message = msg

        elif message['messageLabel'] == 'update':
            server.send_message_to_all(msg)

    def client_left(self, client, server):
        """
        Action taken when a client closes the connection with the server.

        Parameters
        ----------
        client : dict
            The client which just closed the WebSocket connection.
        server : websocket_server.websocket_server.WebsocketServer
            The current server.
        """

        try:
            self.clients.pop(client['id'])
        except:
            print("Error in removing client %s" % client['id'])


class HttpServer(object):
    """
    Encapsulates an HTTP server.

    Attributes
    ----------
    PORT : int
        The port on which to listen.
    HOST : str
        The host used as server.
    address : str
        The full hosting address to connect to the server.
    process : multiprocessing.Process
        The separate process running the server.
    """

    def __init__(self):
        self.PORT = 8000
        self.HOST = '127.0.0.1'
        self.address = 'http://' + self.HOST + ':' + str(self.PORT)
        self.process = self._start_http_process()

    def _start_http_process(self):
        """
        Start the server in a parallel process.

        Returns
        -------
        multiprocessing.Process
            The process running the server.
        """

        service = Process(name='http_server', target=self._start_http_server)
        service.start()

        return service

    def _start_http_server(self):
        """ Start the HTTP server and keep it running. """

        # Go to project root directory.
        os.chdir(ROOT_FOLDER)

        sys.stderr = open(os.path.join(RENDERING_LOGS, "http_stderr.log"), "w")

        httpd = HTTPServer((self.HOST, self.PORT), SimpleHTTPRequestHandler)
        print('\nRendering the environment at : ' + self.HOST + ':' +
              str(self.PORT) + '/' + RENDERING_RELATIVE_PATH + '...\n')

        httpd.serve_forever()