from setuptools import setup

setup(name='gym_anm',
      version='0.0.1',
      install_requires=['gym', 'pandas', 'websocket-client>=0.56.0',
                        'websocket-server==0.4', 'scipy']  #And any other dependencies required
      )