import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import itertools as it
from tqdm import tqdm
from scipy.stats import anglit
import time


def log_likelihood(x, dim, init, boundary=5):
    ''' Return the logarithm of a N-dimensional gaussian likelihood. It is set in such a way that the
    integral of the product with the prior over the parameter space is 1.

    Parameters
    ----------
    x : numpy.array
        If init is set to True, x should be a MxN matrix whose M rows are random N-dimensional vectors.
        In this case it is used to initialize the likelihood of the live points.
        If init is set to False, x should be just a random N dimensional vector.

    dim : int
        Dimension of the parameter space.

    init: bool
        You can choose to use the funcion to initialize the likelihood of the live points (True) or
        to generate just a new likelihood value (False)

    boundary : init, optional
        Boundary of the parameter space.
    '''

    likelihood = []

    if init:
        for v in x:
            exp = v**2
            L = dim*np.log(2*boundary) - 0.5*dim*np.log(2*np.pi) - 0.5*exp.sum()
            likelihood.append(L)
    else:
        L = dim*np.log(2*boundary) - 0.5*dim*np.log(2*np.pi) - 0.5*x.T.dot(x)
        likelihood.append(L)

    return likelihood

def log_prior(x, dim, boundary=5):
    '''Return a uniform prior for each value of the N-dimension vector x. It is set in such a way that the
    integral of the product with the likelihood over the parameter space is 1.

    x : numpy.array
        A random N dimensional vector.

    dim : int
        Dimension of the parameter space.

    boundary : init, optional
        Boundary of the parameter space.
    '''

    prior = []
    mod = np.sqrt(dim) * boundary
    for v in x:
        if np.sqrt((v*v).sum()) > mod:
            prior.append(-np.inf)
        else:
            prior.append(-dim*np.log(2*boundary))

    return prior

def uniform_proposal(x, dim, logLmin, survivor):
    ''' Sample a new object from the prior subject to the constrain L(x_new) > Lworst_old

    Parameters
    ----------
    x : numpy array
        Array (parameter, prior, likelihood) corresponding to the worst likelihood
    logLmin : float64
        Worst likelihood, i.e. The third element of x
    '''
    start = time.time()
    accepted = 0
    rejected = 0
    n = 0
    step = 1
    while True:
        n += 1
        new_line = np.zeros(dim+2, dtype=np.float64)
        new_line[:dim] = np.random.uniform(-step, step, size=dim)
        new_log_prior = log_prior(new_line[:dim], dim)
        new_line[dim] = new_log_prior[0]
        new_line[dim+1] = log_likelihood(new_line[:dim], dim, init=False)[0]

        if new_line[dim+1] < logLmin:
            rejected += 1
            counter += 1
        if new_line[dim+1] > logLmin:
            accepted += 1
            boundary_point[:dim] = new_line[:dim]
            if n > 20:
                end = time.time()
                break
        if accepted != 0 and rejected != 0:
            if accepted > rejected: step *= np.exp(1.0/accepted)
            if accepted < rejected: step /= np.exp(1.0/rejected)

        return new_line, (end-start), accepted, rejected

def normal_proposal(x, dim, logLmin, boundary_point, std):
    ''' Sample a new object from the prior subject to the constrain L(x_new) > Lworst_old

    Parameters
    ----------
    x : numpy array
        Array (parameter, prior, likelihood) corresponding to the worst likelihood
    logLmin : float64
        Worst likelihood, i.e. The third element of x
    '''
    start = time.time()
    accepted = 0
    rejected = 0
    n = 0
    new_line = np.zeros(dim+2, dtype=np.float64)
    multiplier = it.cycle(np.arange(1,5))
    while True:
        n += 1
        k = next(multiplier)
        for i in range(len(new_line[:dim])):
            new_line[:dim][i] = boundary_point[i] + anglit(scale = k*std).rvs()
            #new_line[:dim][i] = np.random.normal(boundary_point[i], std)
            while np.abs(new_line[:dim][i]) > 5.:
                new_line[:dim][i] = boundary_point[i] + anglit(scale = k*std).rvs()
                #new_line[:dim][i] = np.random.normal(boundary_point[i], std)

        new_log_prior = log_prior(new_line[:dim], dim)
        new_line[dim] = new_log_prior[0]

        new_line[dim+1] = log_likelihood(new_line[:dim], dim, init=False)[0]
        if new_line[dim+1] < logLmin:
            rejected += 1
        if new_line[dim+1] > logLmin:
            accepted += 1
            boundary_point[:dim] = new_line[:dim]
            if n > 20:
                end = time.time()
                break

    return new_line, (end-start), accepted, rejected

def nested_samplig(live_points, dim, resample_function=uniform_proposal, verbose=False):
    '''Nested Sampling by Skilling (2006)
    '''

    N = live_points.shape[0]
    f = np.log(0.05)
    area = []; Zlog = [[],[]]; logL_worst = []; T = []; prior_mass = [] # lists for plots

    logZ = -np.inf
    H = 0
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

        H += (np.exp(logwidth)*np.exp(logLw))/np.exp(logZ) * np.log(np.exp(logLw)/np.exp(logZ))
        error = np.sqrt(H/steps)
        Zlog[0].append(np.exp(logZ + error))
        Zlog[1].append(np.exp(logZ - error))

        survivors = np.delete(live_points, Lw_idx, axis=0)
        std = np.mean(np.std(survivors[:dim], axis = 0))
        boundary_point = live_points[Lw_idx,:dim]

        area.append(logwidth+logLw)

        if verbose:
            print("i:{0} d = {1} log(Lw) = {2:.2f} t.c. = {3:.2f} log(Z) = {4:.2f} std = {5:.2f} e = {6:.2f}".format(steps, dim, logLw, (max(live_points[:,dim+1]) - steps/N - f - logZ), logZ, std, error))

        new_sample, t, acc, rej = resample_function(live_points[Lw_idx], dim, logLw, boundary_point, std)
        accepted += acc
        rejected += rej
        T.append(t)

        live_points[Lw_idx] = new_sample
        logwidth -= 1.0/N

        if max(live_points[:,dim+1]) - (steps+steps**0.5)/N < f + logZ:
            break

    final_term = np.log(np.exp(live_points[:,dim+1]).sum()*np.exp(-steps/N)/N)
    logZ = np.logaddexp(logZ, final_term)

    return np.exp(area), Zlog, np.exp(logL_worst), prior_mass, np.exp(logZ), T, steps, accepted, rejected

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Nested sampling')
    parser.add_argument('--dim', '-d', type=int, help='Max dimension of the parameter spaces')
    parser.add_argument('--num_live_points', '-n', type=int, help='Number of live points')
    parser.add_argument('--boundary', '-b', type=int, default=5, help='Boundaries for the prior (centered at zero). The default is 5 ')
    parser.add_argument('--proposal', '-pr', type=int, help='Proposal for the new object from the prior. 0 for uniform, 1 for normal')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the plots')
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
    for d in tqdm(range(1,dim+1,2)):
        live_points = np.zeros((n, d+2), dtype=np.float64) # the first dim columns for each row represents my multidimensional array of parameters
        if args.proposal == 0:
            prop = 'Uniform'
            area_plot, evidence_plot, likelihood_worst, prior_mass, evidence, t_resample, steps, acc, rej = nested_samplig(live_points, d, resample_function=uniform_proposal, verbose=args.verbose)
        if args.proposal == 1:
            prop = 'Normal'
            area_plot, evidence_plot, likelihood_worst, prior_mass, evidence, t_resample, steps, acc, rej = nested_samplig(live_points, d, resample_function=normal_proposal, verbose=args.verbose)

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
            plt.show()
            plt.close('all')

        end = time.time()
        with open(f'results/summaries/{args.proposal}/Summary_{d}.txt', 'w', encoding='utf-8') as file:
            file.write(f'''============ SUMMARY ============
                    \n Dimension of the integral = {d}
                    \n Number of steps required = {steps}
                    \n Evidence = {evidence:.2f}
                    \n Maximum of the likelihood = {(2*boundary/np.sqrt(2*np.pi))**d:.2f}
                    \n Proposal chosen: {prop}
                    \n Last area value = {area_plot[-1]:.2f}
                    \n Last worst Likelihood = {likelihood_worst[-1]}''')
            file.write(f'''\n Accepted and rejected points: {acc}, {rej}
                    \n Mass prior sum = {np.exp(prior_mass).sum():.2f}
                    \n Total time: {end-start:.2f} s
                    \n=================================''')
