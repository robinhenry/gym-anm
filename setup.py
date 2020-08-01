from setuptools import setup

setup(name='gym_anm',
      version='0.1',
      url='http://github.com/robinhenry/gym-anm',
      author='Robin Henry',
      description="A framework to build Reinforcement Learning environments for"
                  " Active Network Management tasks in "
                  "electricity networks.",
      author_email='robin@robinxhenry.com',
      install_requires=['gym', 'pandas', 'websocket-client>=0.56.0',
                        'websocket-server==0.4', 'cvxpy==1.1.1'],
      python_requires='>=3.6',
      )
