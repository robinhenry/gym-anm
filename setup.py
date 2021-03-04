from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gym-anm',
      version='1.0',
      url='http://github.com/robinhenry/gym-anm',
      author='Robin Henry',
      description="A framework to build Reinforcement Learning environments for Active Network Management tasks in electricity networks.",
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT',
      author_email='robin@robinxhenry.com',
      packages=['gym_anm'],
      include_package_data=True,
      install_requires=['numpy', 'cvxpy>=1.1', 'gym', 'pandas', 'websocket-client==0.56.0',
                        'websocket-server==0.4', 'requests'],
      python_requires='>=3.7'
      )
