import gym

import numpy as np

import datetime as dt
import pandas as pd
import argparse
import time
import os

import agents
import networks
import utils.gym


VERBOSE = True
ENVIRONMENTS = {
    'car': 'MountainCarContinuous-v0',
    'dipoc': 'InvertedDoublePendulumPyBulletEnv-v0',
    'hopper': 'HopperPyBulletEnv-v0',
    'pendulum': 'Pendulum-v0',
    'reacher': 'ReacherPyBulletEnv-v0',
    'anm': 'gym_anm:ANM6Easy-v0',
}


def main():

    args = parse_arguments(verbose=VERBOSE)

    # Setup of gym and creation of the environment
    gym.logger.set_level(40)
    env = gym.make(ENVIRONMENTS[args.env])

    # Corrupt the environement with a mask except every `args.masked` steps
    env = utils.gym.EnvMask(env, args.masked)

    if args.render:
        env.render()

    # Agent
    Agent = agents.algos[args.algo]

    agent = Agent(env, args.gamma, args.render, cpu=args.cpu, rnn=args.rnn,
                  rnn_layers=args.rnn_layers, rnn_h_size=args.rnn_h_size,
                  seq_len=args.seq_len)

    # Initialize statistics
    statistics = {
        'train_time': [],
        'train_steps': [],
        'test_score': [],
        'episodes': []
    }
    console_output = 'Episode {:04d}: {:3.2f} ({:.2f}s)'

    total_time = 0.0
    total_steps = 0

    # Training loop
    for e in range(1, args.num_episodes + 1):

        # Train on one epoch
        start = time.time()
        cumulative_reward, num_steps = agent.train(args.num_steps)
        elapsed = time.time() - start

        total_time += elapsed
        total_steps += num_steps

        if VERBOSE:
            print(console_output.format(e, cumulative_reward, elapsed))

        # Evaluate the current policy R times every S episodes
        if e % args.eval_period == 0:
            test_score = evaluate(agent, args.num_rollouts, args.num_steps,
                                  verbose=VERBOSE)
            statistics['test_score'].append(test_score)

            # Register statistics
            statistics['train_steps'].append(total_steps)
            statistics['train_time'].append(total_time)
            statistics['episodes'].append(e)

    env.close()

    # Print statistics
    time_per_frame = total_time / total_steps
    print('Total training time: {:.2f}'.format(total_time))
    print('Total number of frames: {}'.format(total_steps))
    print('Mean training time per frame: {:.3f}'.format(time_per_frame))

    mean_scores = np.mean(np.array(statistics['test_score']), axis=1)
    std_scores = np.std(np.array(statistics['test_score']), axis=1)
    statistics['mean_scores'] = mean_scores
    statistics['std_scores'] = std_scores

    # Save the results
    datetime = dt.datetime.now()
    save_results(args.output, datetime, args.algo, args.rnn, args.rnn_layers,
                 args.rnn_h_size, args.seq_len, time_per_frame, statistics)

    # Save the agent
    if args.save:
        agent.save('saves/' + '{}-{}-{}-{}.pkl'
                   .format(args.algo, args.rnn, args.gamma, args.num_episodes))


def evaluate(agent, num_rollouts, num_steps, verbose=False):
    """
    Evaluate the performance of the agent on the environment using Monte Carlo
    rollouts.

    Arguments
    ---------
    - agent: class implementing a `play` method
        Agent trained on the environment and able to take action in it
    - num_rollouts: int
        Number of rollout to evaluate the policy performance
    - verbose: bool
        Whether to print the mean performance of the policy
    """
    performances = []

    for n in range(num_rollouts):
        performance, _ = agent.eval(num_steps)
        performances.append(performance)

    if verbose:
        mean_scores = np.mean(performances)
        print('Current average performance: {:3.2f}'.format(mean_scores))

    return performances


def save_results(filename, datetime, agent, rnn, rnn_layers, rnn_h_size,
                 seq_len, tpf, statistics):
    """
    Save the results in a csv file, appending the results to other results if
    the file already exists.

    csv format
    ----------
    datetime,agent,rnn,tpf,test_scores,episodes,train_steps,train_time,
    """
    columns = ['agent', 'rnn', 'rnn_layers', 'rnn_h_size', 'seq_len', 'tpf',
               'mean_scores', 'std_scores', 'episodes', 'train_steps',
               'train_time']

    os.makedirs('results', exist_ok=True)
    filepath = 'results/{}.csv'.format(filename)
    if os.path.isfile(filepath):
        results = pd.read_csv(filepath, index_col='datetime')
    else:
        results = pd.DataFrame(columns=columns)
        results.index.name = 'datetime'

    base_column = [
        agent,
        rnn,
        rnn_layers if rnn else None,
        rnn_h_size if rnn else None,
        seq_len if rnn else None,
        tpf,
        list(statistics['mean_scores']),
        list(statistics['std_scores']),
        list(statistics['episodes']),
        list(statistics['train_steps']),
        list(statistics['train_time'])
    ]

    results.loc[datetime, columns] = base_column
    results.to_csv(filepath)


def parse_arguments(verbose=False):
    """
    Parse the arguments from the stdin
    """
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env', type=str, help='name of the environment',
                        default='dipoc', choices=ENVIRONMENTS.keys())
    parser.add_argument('--masked', type=int, help='state if visible once '
                        'every MASKED frame', default=1)

    # User preferences
    parser.add_argument('-r', '--render', help='render the game during '
                        'training', action='store_true')
    parser.add_argument('--cpu', help='train on CPU', action='store_true')

    # Agent
    parser.add_argument('algo', type=str, help='name of the agent algorithm',
                        choices=agents.algos.keys())
    parser.add_argument('--gamma', help='discount factor', type=float,
                        default=.99)
    parser.add_argument('--rnn', type=str, help='type of rnn', default=None,
                        choices=networks.rnns.keys())

    # Episodes
    parser.add_argument('-M', '--num-episodes', help='number of episodes',
                        type=int, default=400)
    parser.add_argument('-T', '--num-steps', help='maximum number of '
                        'transitions per episode', type=int, default=1000)
    parser.add_argument('-S', '--eval-period', help='evaluation period in '
                        'number of episodes', type=int, default=10)
    parser.add_argument('-R', '--num_rollouts', help='number of rollouts to '
                        'evaluate performance', type=int, default=20)

    # Outputs
    parser.add_argument('-o', '--output', help='file name for the results',
                        default='results', type=str)
    parser.add_argument('-s', '--save', help='whether to save the agent',
                        action='store_true')

    # RNN parameters
    parser.add_argument('-H', '--rnn-h-size', help='hidden size in each RNN '
                        'layer', type=int, default=256)
    parser.add_argument('-L', '--rnn-layers', help='Number of RNN layers',
                        type=int, default=2)
    parser.add_argument('-N', '--seq-len', help='Length of the sequences used '
                        'to train the RNN', type=int, default=8)

    args = parser.parse_args()

    if verbose:
        print('\nArguments\n---------')
        for arg, val in sorted(vars(args).items()):
            print('{}: {}'.format(arg, val))
        print()

    return args


if __name__ == '__main__':
    main()
