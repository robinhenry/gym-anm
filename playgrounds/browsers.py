import time
from gym_anm.envs import ANM6Easy
import os
import sys
from gym_anm.envs.anm6_env.rendering.py.servers import WsServer
from gym_anm import ROOT_FOLDER
import webbrowser
import requests


def start_http_server():
    ws_server = WsServer()

    # Wait until server is fully started.
    while True:
        request = requests.get(a, timeout=0.01)
        if request.status_code == 200:
            break

    webbrowser.open_new_tab(a)
    print('Done!')


if __name__ == '__main__':
    start_http_server()


    # env = ANM6Easy()
    # env.reset()
    #
    # for i in range(100):
    #     env.step(env.action_space.sample())
    #     env.render()
    #     time.sleep(2)
