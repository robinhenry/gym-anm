from multiprocessing import Process
from http.server import HTTPServer, SimpleHTTPRequestHandler


def start_http_server():
    service = Process(name='http_server', target=_start_http_server)
    service.start()
    return service

def _start_http_server():
    PORT = 8000
    HOST = '0.0.0.0'

    httpd = HTTPServer((HOST, PORT), SimpleHTTPRequestHandler)
    print('Serving at : ' + HOST + ':' + str(PORT) + '...')
    httpd.serve_forever()
