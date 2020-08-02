from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gym_anm',
      version='0.0.1',
      url='http://github.com/robinhenry/gym-anm',
      author='Robin Henry',
      description="A framework to build Reinforcement Learning environments for Active Network Management tasks in electricity networks.",
      long_description=long_description,
      long_description_content_type='text/markdown',
      author_email='robin@robinxhenry.com',
      packages=['gym_anm'],
      install_requires=['gym', 'pandas', 'websocket-client==0.56.0',
                        'websocket-server==0.4', 'cvxpy>=1.1.1', 'requests'],
      python_requires='>=3.6'
      )
