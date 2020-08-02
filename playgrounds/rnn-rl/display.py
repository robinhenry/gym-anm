import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os


if len(sys.argv) < 2:
    print('USAGE: ./{} RESULT_FILENAME'.format(sys.argv[0]))
    sys.exit(1)

filename = sys.argv[1]
filepath = 'results/{}.csv'.format(filename)

if not os.path.exists(filepath):
    raise FileNotFoundError('The results file has not been found')


def list_converter(x):
    return np.array(eval(x))


to_convert = ['episodes', 'train_steps',
              'train_time', 'mean_scores', 'std_scores']

results = pd.read_csv(filepath, index_col='datetime', converters={
                      x: list_converter for x in to_convert})


statistics = ('episodes', 'train_steps', 'train_time')
labels = ('Episodes [-]', 'Frames [-]', 'Training time [s]')

for statistic, label in zip(statistics, labels):
    fig, ax = plt.subplots(figsize=(8, 6))

    for index, result in results.iterrows():

        ref = ' '.join((result['agent'], result['rnn']
                        if not pd.isna(result['rnn']) else ''))

        ax.plot(result[statistic], result['mean_scores'], label=ref)

        lower_bound = result['mean_scores'] - result['std_scores']
        upper_bound = result['mean_scores'] + result['std_scores']
        ax.fill_between(result[statistic], lower_bound, upper_bound, alpha=.2)

        ax.set_xlabel(label)
        ax.set_ylabel('Cumulative reward')

    ax.legend()
    os.makedirs('plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig('plots/{}_{}.pdf'.format(filename, statistic),
                transparent=True)
    plt.show()
