import os.path as path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='talk', style='white')


def main():
    file1 = path.join('data', 'state_dep_trust.pkl')
    with open(file1, 'rb') as f:
        data1 = pickle.load(f)

    trust_all1 = np.array(list(data1.values()))
    trust_all1 = trust_all1.reshape((-1, 10))
    mean = trust_all1.mean(axis=0)
    std = trust_all1.std(axis=0)
    ci = 1.96 * std / np.sqrt(trust_all1.shape[0])

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(mean) + 1), mean, lw=2, label='state dep', c='tab:blue')
    ax.fill_between(np.arange(1, len(mean) + 1), mean - ci, mean + ci, color='tab:blue', alpha=0.5)
    ax.grid(axis='y')

    file2 = path.join('data', 'const_trust.pkl')
    with open(file2, 'rb') as f:
        data2 = pickle.load(f)

    trust_all2 = np.array(list(data2.values()))
    trust_all2 = trust_all2.reshape((-1, 10))
    mean = trust_all2.mean(axis=0)
    std = trust_all2.std(axis=0)
    ci = 1.96 * std / np.sqrt(trust_all2.shape[0])
    ax.plot(np.arange(1, len(mean) + 1), mean, lw=2, label='constant', c='tab:orange')
    ax.fill_between(np.arange(1, len(mean) + 1), mean - ci, mean + ci, color='tab:orange', alpha=0.5)
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
