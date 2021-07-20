'''Python implementation of Nested Sampling (Skilling 2004) in order to compute the integral of a N-dim gaussian'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import logging
import time
from nested_sampling.functions import log_likelihood, autocorrelation, proposal

def nested_samplig(live_points, dim, boundary, proposal_distribution, verbose=False):
    '''Nested Sampling by Skilling (2004)

    Parameters
    ----------
    live_points : numpy array
        Numpy array of dimension (number of live points, dimension + 2). From column 0
        to d(=dimension) the vectors of parameter sampled from the parameter space are
        placed, column d+1 corresponds to the priors value and the last to the likelihood values
    dim : int
        Dimension of the parameter space.
    resample_function : str
        Choose between uniform or normal resample distribution
    verbose : bool
        Print some information. The default is False

    Returns
    -------
    area : numpy array
        Array of the area elements accumulated during iterations
    Zlog : list
        List of two list: logZ + error and logZ - error. They are used for plots
    logL_worst : list
        List of the worst likelihoods obtained during iterations. They are the values that
        define the profile of the likelihood used in the actual computation of the Riemann sum
    prior_mass : list
        List of the value of the prior mass accumulated during the iteration
    T : list
        List of times required for sample the new objects with the proposal function
    steps : int
        Number of steps required
    accepted, rejected : int
        Accepted and rejected points over the whole NS run. They are the sum of all the accepted/rejected
        points during the sampling of the new object in the proposal function
    logH_list : list
        List of information values accumulated during iterations
    error : int
        Estimated error on the evidence. It is a first approximation based on the standard deviation
        of the prior mass values (the one found statistically, see Nested Sampling (Skilling 2004))

        '''

    N = live_points.shape[0]
    f = np.log(0.01)
    area = []; Zlog = [[],[]]; logL_worst = []; T = []; prior_mass = []; logH_list = []

    logZ = -np.inf
    logH = -np.inf
    parameters = np.random.uniform(-boundary, boundary, size=(N, dim))
    live_points[:, :dim] = parameters
    live_points[:, dim] = log_likelihood(parameters, dim, init=True)
    logwidth = np.log(1.0 - np.exp(-1.0/N))

    steps = 0
    rejected = 0
    accepted = 0
    while True:
        steps += 1
        prior_mass.append(logwidth)

        Lw_idx = np.argmin(live_points[:, dim])
        logLw = live_points[Lw_idx, dim]
        logL_worst.append(logLw)

        logZnew = np.logaddexp(logZ, logwidth+logLw)
        logZ = logZnew

        logH = np.logaddexp(logH, logwidth + logLw - logZ + np.log(logLw - logZ))
        logH_list.append(logH)
        error = np.sqrt(np.exp(logH)/N)
        Zlog[0].append(logZ + error)
        Zlog[1].append(logZ - error)

        survivors = np.delete(live_points, Lw_idx, axis=0)
        std = np.mean(np.std(survivors[:dim], axis = 0))
        boundary_point = live_points[Lw_idx,:dim]

        area.append(logwidth+logLw)

        if verbose:
            print("i:{0} d={1} log(Lw)={2:.2f} term={3:.2f} log(Z)={4:.2f} std={5:.2f} e={6:.2f} H={7:.2f} prop={8}".format(steps, dim, logLw, (max(live_points[:,dim]) - steps/N - f - logZ), logZ, std, error, np.exp(logH), proposal_distribution))

        new_sample, t, acc, rej = proposal(live_points[Lw_idx], dim, boundary, std, proposal_distribution)
        accepted += acc
        rejected += rej
        T.append(t)

        live_points[Lw_idx] = new_sample
        logwidth -= 1.0/N

        if max(live_points[:,dim]) - (steps+steps**0.5)/N < f + logZ:
            break

    final_term = np.log(np.exp(live_points[:,dim]).sum()*np.exp(-steps/N)/N)
    logZ = np.logaddexp(logZ, final_term)
    area = np.exp(area)

    return area, Zlog, logL_worst, prior_mass, logZ, T, steps, accepted, rejected, logH_list, error

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Nested sampling')
    parser.add_argument('--dim', '-d', type=int, help='Max dimension of the parameter spaces')
    parser.add_argument('--num_live_points', '-n', type=int, help='Number of live points')
    parser.add_argument('--boundary', '-b', type=int, default=5, help='Boundaries for the prior (centered at zero). The default is 5 ')
    parser.add_argument('--proposal', '-pr', type=int, help='Proposal for the new object from the prior. 0 for uniform, 1 for normal')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the plots')
    parser.add_argument('--summary_plot', '-sp', action='store_true', help='Plot some summary information')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print some info during iterations. the default is False')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))

    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level=levels[args.log])
    np.random.seed(95)

    n = args.num_live_points
    dim = args.dim
    boundary = args.boundary
    if args.proposal == 0: prop = 'uniform'
    if args.proposal == 1: prop = 'normal'

    time_tot = []; log_evidence_values = []; error_values = []

    range_dim = np.arange(2,dim+1)

    true_values = [-d*np.log(2*boundary) for d in range_dim]


    for d in tqdm(range_dim):
        t_start = time.time()

        live_points = np.zeros((n, d+1), dtype=np.float64)
        area_plot, evidence_plot, likelihood_worst, prior_mass, log_evidence, t_resample, steps, acc, rej, logH, error = nested_samplig(live_points, d, boundary, proposal_distribution=prop, verbose=args.verbose)

        t_end = time.time()
        t_total = t_end - t_start

        time_tot.append(t_total)
        log_evidence_values.append(log_evidence)
        error_values.append(error)

        if args.plot:

            fig, ax1 = plt.subplots()
            ax1.plot(area_plot, color = 'k', label = 'Li*wi')
            ax2 = ax1.twinx()
            ax2.plot(evidence_plot[0], color = 'r', linewidth=0.6, label = 'logZ +- e')
            ax2.plot(evidence_plot[1], color = 'r', linewidth=0.6)
            plt.fill_between(np.arange(len(evidence_plot[0])),evidence_plot[0],evidence_plot[1], color='r', alpha=0.3)
            ax1.set_xlabel('Iterations')
            plt.grid()
            fig.legend()
            plt.savefig(f'results/images/{args.proposal}/area_dynamics_{d}.png')

            plt.figure()
            plt.scatter(np.arange(len(t_resample)),t_resample, s=0.5, c='k')
            plt.yscale('log')
            plt.xlabel('Iterations')
            plt.ylabel('Resampling time')
            plt.title('Time for resampling')
            plt.savefig(f'results/images/{args.proposal}/resampling_time_{d}.png')

            plt.figure()
            plt.plot(prior_mass, logH, 'k')
            plt.xlabel('Prior mass (log)')
            plt.ylabel('logH')
            plt.title('Information')
            plt.savefig(f'results/images/{args.proposal}/information_{d}.png')
            plt.close('all')

        with open(f'results/summaries/{args.proposal}/Summary_{d}.txt', 'w', encoding='utf-8') as file:
            file.write(f'''============ SUMMARY ============
                    \n Dimension of the integral = {d}
                    \n Number of steps required = {steps}
                    \n Evidence = {log_evidence:.2f} +- {error:.2f}
                    \n Theoretical value = -{d*np.log(2*boundary)}
                    \n Information = {np.exp(logH)[-1]:.2f}
                    \n Maximum of the likelihood = {-d*np.log(np.sqrt(2*np.pi)):.2f}
                    \n Proposal chosen: {prop}
                    \n Last area value = {area_plot[-1]:.2f}
                    \n Last worst Likelihood = {likelihood_worst[-1]:.2f}
                    \n Accepted and rejected points: {acc}, {rej}
                    \n Mass prior sum = {np.exp(prior_mass).sum():.2f}
                    \n Total time: {t_total:.2f} s
                    \n=================================''')
            file.close()

    compared_results = pd.DataFrame({'log_evidence_values': log_evidence_values, 'error': error_values, 'time': time_tot})
    compared_results.to_csv(f'results/results_{args.proposal}.csv', index=False)

    if args.summary_plot:

        plt.figure()
        plt.plot(range_dim, time_tot, 'k--')
        plt.ylabel('Time (s)')
        plt.xlabel('Dimension')
        plt.grid()
        plt.scatter(range_dim, time_tot, c='black')
        plt.savefig(f'results/images/Total_time_per_dim_{args.proposal}.png')

        plt.figure()
        plt.plot(range_dim, true_values, 'k--', linewidth=0.6, alpha=0.4)
        plt.scatter(range_dim, true_values, s=3, c='black', label='True Values')
        plt.errorbar(range_dim, log_evidence_values, yerr=error_values, linewidth=0.6, elinewidth=1, linestyle='-.', label='Computed Values')
        plt.xlabel('Dimension')
        plt.ylabel('logZ')
        plt.legend()
        plt.savefig(f'results/images/Results_{args.proposal}.png')
        plt.close('all')
