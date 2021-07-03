import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import time


def log_likelihood(x, dim, init, boundary=10):

    likelihood = []

    if init:
        for v in x:
            exp = v**2
            L = np.log(((2*boundary) / np.sqrt(2*np.pi))**dim * np.exp(-0.5*exp.sum()))
            likelihood.append(L)
    else:
        L = np.log(((2*boundary) / np.sqrt(2*np.pi))**dim * np.exp(-0.5*x.T.dot(x)))
        likelihood.append(L)

    return likelihood

def log_prior(x, dim, boundary=10):

    prior = []
    mod = np.sqrt(dim) * boundary
    for v in x:
        if np.sqrt((v*v).sum()) > mod:
            prior.append(-np.inf)
        else:
            prior.append(np.log(1 / (2*boundary)**dim))

    return prior

def uniform_proposal(x, dim, logLmin):
    ''' Sample a new object from the prior subject to the constrain L(x_new) > Lworst_old

    Parameters
    ----------
    x : numpy array
        Array (parameter, prior, likelihood) corresponding to the worst likelihood
    logLmin : float64
        Worst likelihood, i.e. The third element of x
    '''
    start = time.time()
    counter = 0
    a = 0
    while True:
        counter += 1
        if counter > 200:
            a += 1
            counter = 0
        new_line = np.zeros(dim+2, dtype=np.float64)
        new_line[:dim] = np.random.uniform(-10 + 10*a/1000, 10 - 10*a/1000, size=dim)
        new_log_prior = log_prior(new_line[:dim], dim)
        new_line[dim] = new_log_prior[0]
        # acceptance MH rule
        # if (new_log_prior - x[:dim]).any() > np.log(np.random.uniform(0, 1)):
        new_line[dim+1] = log_likelihood(new_line[:dim], dim, init=False)[0]
        new_log_likelihood = new_line[dim+1]
        if new_log_likelihood > logLmin:  # check if the new likelihood is greater then the old one
            end = time.time()
            t = end-start
            print('Time for resampling: {0:.2f} s'.format(t))
            return new_line

def nested_samplig(live_points,dim, steps, resample_function=uniform_proposal):
    N = live_points.shape[0]
    Area = []; Zlog = []; logL_worst = []

    logZ = -np.inf
    parameters = np.random.uniform(-10, 10, size=(N, dim))
    live_points[:, :dim] = parameters
    live_points[:, dim] = log_prior(parameters, dim)
    live_points[:, dim+1] = log_likelihood(parameters, dim, init=True)
    logwidth = np.zeros(steps+1)
    logwidth[0] = np.log(1.0 - np.exp(-1.0/N))
    for i in range(steps):
        Lw_idx = np.argmin(live_points[:, dim+1])
        logLw = live_points[Lw_idx, dim+1]
        logZnew = np.logaddexp(logZ, logwidth[i]+logLw)

        logL_worst.append(logLw)
        Area.append(logwidth[i]+logLw)
        Zlog.append(logZnew)

        logZ = logZnew
        print("n:{0} logL_worst = {1:.5f} --> width = {2:.5f} Z = {3:.5f}".format(i,
                                                                            np.exp(logLw), logwidth[i], np.exp(logZ)))
        new_sample = resample_function(live_points[Lw_idx], dim, logLw)
        live_points[Lw_idx] = new_sample
        logwidth[i+1] = logwidth[i] - 1.0/N
    return np.exp(Area), np.exp(Zlog), np.exp(logL_worst), logwidth, np.exp(logZ)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Nested sampling')
    parser.add_argument('--dim', '-d', type=int, help='Dimension of the parameter space')
    parser.add_argument('--num_live_points', '-n', type=int, help='Number of live points')
    parser.add_argument('--steps', '-s', type=int, help='Number steps')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))

    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level=levels[args.log])


    n = args.num_live_points
    dim = args.dim
    live_points = np.zeros((n, dim+2), dtype=np.float64) # the first dim columns for each row represents my multidimensional array of parameters
    area_plot, evidence_plot, likelihood_worst, prior_mass, evidence = nested_samplig(live_points, dim, steps=args.steps, resample_function=uniform_proposal)

    plt.figure()
    plt.plot(area_plot)
    plt.xlabel('Iterations')
    plt.ylabel('Areas Li*wi')
    plt.title('Dynamics of the Area')

    plt.figure()
    plt.plot(evidence_plot)
    plt.xlabel('Iterations')
    plt.ylabel('Evidence Z')
    plt.title('Evidence as functioon of iterations')

    X = []
    plt.figure()
    plt.scatter(prior_mass[:len(likelihood_worst)], likelihood_worst, s=0.1)
    plt.xlabel('log(X)')
    plt.ylabel('Worst Likelihood')

    print('Evidence = {0:.5f}'.format(evidence))
    print('Check of priors sum: ', np.sum(np.exp(prior_mass)))
    plt.show()
