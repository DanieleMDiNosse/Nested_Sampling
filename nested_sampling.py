import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from scipy.stats import anglit
import time
from functions import log_likelihood, log_prior, autocorrelation


def proposal(x, dim, logLmin, boundary_point, std, distribution):
    ''' Sample a new object from the prior subject to the constrain L(x_new) > Lworst_old

    Parameters
    ----------
    x : numpy array
        Array (parameter, prior, likelihood) corresponding to the worst likelihood
    dim : int
        Dimension of the parameter space.
    logLmin : float64
        Worst likelihood, i.e. The third element of x
    boundary_points : numpy array
        Array of parameters corresponding to the worst likelihood computer at iteration
        k of NS
    std : float
        Limits of the uniform distribution proposal or standard deviation of the normal/anglit
        distribution. The name comes from the fact that it correspondsto the mean of standard
        deviations of the points along the axis of the parameter space
    distribution : string
        Choose the distribution from which the new object should be sampled.
        Available options are 'uniform', 'normal', 'anglit'.
    Returns
    -------
    new_line : numpy array
        New sampled object that satisfies the likelihood constrain
    t : float
        Seconds required for the resampling
    accepted, rejected : int
        Accepted/rejected number of points during the resampling
    '''
    start = time.time()
    accepted = 0
    rejected = 0
    n = 0
    step = 1
    while True:
        n += 1
        new_line = np.zeros(dim+2, dtype=np.float64)
        k_u = 1.45/np.log(np.sqrt(dim))
        k_n = 1
        k_a = 1.45/np.log(np.sqrt(dim))

        if distribution == 'uniform':
            new_line[:dim] = boundary_point[:dim] + np.random.uniform(-k_u*std, k_u*std, size=dim)


        for i in range(len(new_line[:dim])):
            if distribution == 'normal':
                new_line[:dim][i] = np.random.normal(boundary_point[i], k_n*std)
                while np.abs(new_line[:dim][i]) > 5.:
                    new_line[:dim][i] = np.random.normal(boundary_point[i], k_n*std)

            if distribution == 'anglit':
                new_line[:dim][i] = boundary_point[i] + anglit(scale=k_a*std).rvs()
                while np.abs(new_line[:dim][i]) > 5.:
                    new_line[:dim][i] = boundary_point[i] + anglit(scale=k_a*std).rvs()

        new_log_prior = log_prior(new_line[:dim], dim)
        new_line[dim] = new_log_prior[0]
        new_line[dim+1] = log_likelihood(new_line[:dim], dim, init=False)[0]

        if new_line[dim+1] < logLmin:
            rejected += 1
        if new_line[dim+1] > logLmin:
            accepted += 1
            boundary_point[:dim] = new_line[:dim]
            if n > 10:
                end = time.time()
                t = end - start
                break
        if accepted != 0 and rejected != 0:
            if accepted > rejected: std *= np.exp(1.0/accepted)
            if accepted < rejected: std /= np.exp(1.0/rejected)

    return new_line, t, accepted, rejected

def nested_samplig(live_points, dim, proposal_distribution, verbose=False):
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
        '''

    N = live_points.shape[0]
    f = np.log(0.01)
    area = []; Zlog = [[],[]]; logL_worst = []; T = []; prior_mass = []; logH_list = []

    logZ = -np.inf
    logH = -np.inf
    parameters = np.random.uniform(-5, 5, size=(N, dim))
    live_points[:, :dim] = parameters
    live_points[:, dim] = log_prior(parameters, dim)
    live_points[:, dim+1] = log_likelihood(parameters, dim, init=True)
    logwidth = np.log(1.0 - np.exp(-1.0/N))

    steps = 0
    rejected = 0
    accepted = 0
    while True:
        steps += 1
        prior_mass.append(logwidth)

        Lw_idx = np.argmin(live_points[:, dim+1])
        logLw = live_points[Lw_idx, dim+1]
        logL_worst.append(logLw)

        logZnew = np.logaddexp(logZ, logwidth+logLw)
        logZ = logZnew

        logH = np.logaddexp(logH, logwidth + logLw - logZ + np.log(logLw - logZ))
        logH_list.append(logH)
        error = np.sqrt(np.exp(logH)/steps)
        Zlog[0].append(np.exp(logZ + error))
        Zlog[1].append(np.exp(logZ - error))

        survivors = np.delete(live_points, Lw_idx, axis=0)
        std = np.mean(np.std(survivors[:dim], axis = 0))
        boundary_point = live_points[Lw_idx,:dim]

        area.append(logwidth+logLw)

        if verbose:
            print("i:{0} d={1} log(Lw)={2:.2f} term={3:.2f} log(Z)={4:.2f} std={5:.2f} e={6:.2f} H={7:.2f} prop={8}".format(steps, dim, logLw, (max(live_points[:,dim+1]) - steps/N - f - logZ), logZ, std, error, np.exp(logH), proposal_distribution))

        new_sample, t, acc, rej = proposal(live_points[Lw_idx], dim, logLw, boundary_point, std, proposal_distribution)
        accepted += acc
        rejected += rej
        T.append(t)

        live_points[Lw_idx] = new_sample
        logwidth -= 1.0/N

        if max(live_points[:,dim+1]) - (steps+steps**0.5)/N < f + logZ:
            break

    final_term = np.log(np.exp(live_points[:,dim+1]).sum()*np.exp(-steps/N)/N)
    logZ = np.logaddexp(logZ, final_term)
    area = np.exp(area)
    logL_worst = np.exp(logL_worst)
    logZ = np.exp(logZ)

    return area, Zlog, logL_worst, prior_mass, logZ, T, steps, accepted, rejected, logH_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Nested sampling')
    parser.add_argument('--dim', '-d', type=int, help='Max dimension of the parameter spaces')
    parser.add_argument('--num_live_points', '-n', type=int, help='Number of live points')
    parser.add_argument('--boundary', '-b', type=int, default=5, help='Boundaries for the prior (centered at zero). The default is 5 ')
    parser.add_argument('--proposal', '-pr', type=int, help='Proposal for the new object from the prior. 0 for uniform, 1 for normal, 2 for anglit')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the plots')
    parser.add_argument('--total_time_plot', '-t', action='store_true', help='Plot the total time for each dimension')
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

    start = time.time()
    n = args.num_live_points
    dim = args.dim
    boundary = args.boundary
    time_tot = []
    if args.proposal == 0: prop = 'uniform'
    if args.proposal == 1: prop = 'normal'
    if args.proposal == 2: prop = 'anglit'

    for d in tqdm(range(2,dim+1,3)):
        t_start = time.time()
        live_points = np.zeros((n, d+2), dtype=np.float64)

        area_plot, evidence_plot, likelihood_worst, prior_mass, evidence, t_resample, steps, acc, rej, logH = nested_samplig(live_points, d, proposal_distribution = prop, verbose=args.verbose)

        t_end = time.time()
        t_total = t_end - t_start
        time_tot.append(t_total)
        #print(logH[-1])
        #plt.figure()
        #plt.plot(prior_mass[:len(area_plot)],area_plot)
        #plt.xlabel('Prior mass (log)')
        #plt.ylabel('L*w')
        #plt.show()
        if args.plot:

            fig, ax1 = plt.subplots()
            ax1.plot(area_plot, color = 'k', label = 'Li*wi')
            ax2 = ax1.twinx()
            ax2.plot(evidence_plot[0], color = 'r', label = 'Z + e')
            ax2.plot(evidence_plot[1], color = 'r', label = 'Z - e')
            plt.fill_between(np.arange(len(evidence_plot[0])),evidence_plot[0],evidence_plot[1], color='r', alpha=0.3)
            ax1.set_xlabel('Iterations')
            plt.grid()
            fig.legend()
            plt.savefig(f'results/images/{args.proposal}/area_dynamics_{d}.png')

            plt.figure()
            plt.scatter(prior_mass[:len(likelihood_worst)], likelihood_worst, s=0.1, c='k')
            plt.xlabel('log(X)')
            plt.ylabel('Worst L')
            plt.savefig(f'results/images/{args.proposal}/worst_likelihood_{d}.png')
            plt.fill_between(prior_mass[:len(likelihood_worst)],likelihood_worst, color='k', alpha=0.3)

            plt.figure()
            plt.scatter(np.arange(len(t_resample)),t_resample, s=0.5, c='k')
            plt.yscale('log')
            plt.xlabel('Iterations')
            plt.ylabel('Resampling time')
            plt.title('Time for resampling')
            plt.savefig(f'results/images/{args.proposal}/resampling_time_{d}.png')
            plt.close('all')

        end = time.time()
        with open(f'results/summaries/{args.proposal}/Summary_{d}.txt', 'w', encoding='utf-8') as file:
            file.write(f'''============ SUMMARY ============
                    \n Dimension of the integral = {d}
                    \n Number of steps required = {steps}
                    \n Evidence = {evidence:.2f} +- {np.sqrt(np.exp(logH)/steps)}
                    \n Information = {np.exp(logH)}
                    \n Maximum of the likelihood = {(2*boundary/np.sqrt(2*np.pi))**d:.2f}
                    \n Proposal chosen: {prop}
                    \n Last area value = {area_plot[-1]:.2f}
                    \n Last worst Likelihood = {likelihood_worst[-1]}
                    \n Accepted and rejected points: {acc}, {rej}
                    \n Mass prior sum = {np.exp(prior_mass).sum():.2f}
                    \n Total time: {end-start:.2f} s
                    \n=================================''')

    if args.total_time_plot:
        plt.figure()
        plt.plot(time_tot, 'k--')
        plt.ylabel('Time (s)')
        plt.xlabel('Dimension')
        plt.grid()
        plt.scatter(np.arange(len(time_tot)), time_tot, c='black')
        plt.savefig(f'results/images/Total_time_per_dim.png')
        plt.close('all')
