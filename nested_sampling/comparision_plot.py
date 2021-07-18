import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


def comparision_plot(range_dim, true_values, proposal=[0,1], folder_values_path='results/'):
    '''Plot the logarithm of the evidence for dimension in range_dim and the total time
    required to complete the integration

    Parameters
    ----------
    folder_values_path : str, optional
        Path of the folder. The default is 'results/'
    proposal : list, optional
        List of the proposal distribution using the notation of nested_sampling.py:
        0 for uniform and 1 for normal. The default is [0,1]
    true_values : list
        List of the theoretical values of the integral for the evidence
    range_dim : numpy array
        Array of dimension used for the integration in nested_sampling.py

    Returns
    --------
    None
    '''
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for prop in proposal:
        values = pd.read_csv(folder_values_path + f'results_{prop}.csv')
        log_values = values['log_evidence_values']
        error = values['error']
        time = values['time']

        ax1.errorbar(range_dim, log_values, yerr=error, linewidth=0.6, elinewidth=1, linestyle='-.', label=f'Computed Values {prop}')
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('logZ')

        ax2.scatter(range_dim, time, label=f'{prop}')
        ax2.plot(range_dim, time, 'k--', linewidth=0.6, alpha=0.4)
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Time (s)')

    ax1.plot(range_dim, true_values, 'k--', linewidth=0.6, alpha=0.4)
    ax1.scatter(range_dim, true_values, s=3, c='black', label='True Values')
    fig1.savefig(f'results/images/Comparision_evidence_{proposal}.png')
    fig2.savefig(f'results/images/Comparision_time_{proposal}.png')
    fig1.legend()
    fig2.legend()
    plt.show()
    plt.close('all')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Nested sampling')
    parser.add_argument('--dim', '-d', type=int, help='Max dimension of the parameter spaces')
    parser.add_argument('--boundary', '-b', type=int, default=5, help='Boundaries for the prior (centered at zero). The default is 5 ')
    args = parser.parse_args()


    dim = args.dim
    boundary = args.boundary
    range_dim = np.arange(2,dim+1,3)
    true_values = [-d*np.log(2*boundary) for d in range_dim]
    proposal = [0,1]
    folder_values_path = 'results/'

    comparision_plot(range_dim, true_values)
