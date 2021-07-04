import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import time


def log_likelihood(x, dim, init, boundary=10):
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
            L = np.log(((2*boundary) / np.sqrt(2*np.pi))**dim * np.exp(-0.5*exp.sum()))
            likelihood.append(L)
    else:
        L = np.log(((2*boundary) / np.sqrt(2*np.pi))**dim * np.exp(-0.5*x.T.dot(x)))
        likelihood.append(L)

    return likelihood

def log_prior(x, dim, boundary=10):
    '''Return a uniform prior for each value of the N-dimension vector x. It is set in such a way that the
    integral of the product with the likelihood over the parameter space is 1.

    x : numpy.array
        A random N dimensional vector.

    dim : int
        Dimension of the parameter space.

    init: bool
        You can choose to use the funcion to initialize the likelihood of the live points (True) or
        to generate just a new likelihood value (False)

    boundary : init, optional
        Boundary of the parameter space.
    '''

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
    shrink = 0
    while True:
        counter += 1
        if counter > 200:
            shrink += 1
            counter = 0
        new_line = np.zeros(dim+2, dtype=np.float64)
        #new_line[:dim] = np.random.normal(survivor, 0.01, size=dim)
        new_line[:dim] = np.random.uniform(-10 + 10*shrink/700, 10 - 10*shrink/700, size=dim)
        new_log_prior = log_prior(new_line[:dim], dim)
        new_line[dim] = new_log_prior[0]
        # acceptance MH rule
        if (new_log_prior - x[:dim]).any() > np.log(np.random.uniform(0, 1)):
            new_line[dim+1] = log_likelihood(new_line[:dim], dim, init=False)[0]
            new_log_likelihood = new_line[dim+1]
            if new_log_likelihood > logLmin:  # check if the new likelihood is greater then the old one
                end = time.time()
                t = end-start
                print('Time for resampling: {0:.2f} s'.format(t))
                return new_line, t, shrink

def nested_samplig(live_points,dim, resample_function=uniform_proposal):
    '''Nested Sampling by Skilling (2006)
    '''

    N = live_points.shape[0]
    f = np.log(0.00001)
    Area = []; Zlog = []; logL_worst = []; T = []; A = [] # lists for plots

    logZ = -np.inf
    parameters = np.random.uniform(-10, 10, size=(N, dim))
    live_points[:, :dim] = parameters
    live_points[:, dim] = log_prior(parameters, dim)
    live_points[:, dim+1] = log_likelihood(parameters, dim, init=True)
    #logwidth = np.zeros(steps+1)
    logwidth = np.log(1.0 - np.exp(-1.0/N))
    prior_mass = 0
    #for i in range(steps):
    i = 0
    while True:
        prior_mass += np.exp(logwidth)

        Lw_idx = np.argmin(live_points[:, dim+1])
        logLw = live_points[Lw_idx, dim+1]
        logZnew = np.logaddexp(logZ, logwidth+logLw)

        #survivors = np.delete(live_points, Lw_idx, axis=0)
        #k = survivors.shape[0]
        #survivor = live_points[int(np.random.uniform(k)),:dim]

        logL_worst.append(logLw)
        Area.append(logwidth+logLw)
        Zlog.append(logZnew)

        logZ = logZnew
        print("n:{0} L_worst = {1:.5f} --> width = {2:.5f} Z = {3:.5f}".format(i,
                                                                            np.exp(logLw), np.exp(logwidth), np.exp(logZ)))
        new_sample, t, a = resample_function(live_points[Lw_idx], dim, logLw)
        A.append(a)
        T.append(t)
        live_points[Lw_idx] = new_sample
        logwidth -= 1.0/N
        if -0.5*dim - i/N < f + logZ:
            break
        i += 1
    return np.exp(Area), np.exp(Zlog), np.exp(logL_worst), prior_mass, np.exp(logZ), T, A

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Nested sampling')
    parser.add_argument('--dim', '-d', type=int, help='Dimension of the parameter space')
    parser.add_argument('--num_live_points', '-n', type=int, help='Number of live points')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot the plots')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))

    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level=levels[args.log])

    start = time.time()
    n = args.num_live_points
    dim = args.dim
    live_points = np.zeros((n, dim+2), dtype=np.float64) # the first dim columns for each row represents my multidimensional array of parameters
    area_plot, evidence_plot, likelihood_worst, prior_mass, evidence, t_resample, prior_shrink = nested_samplig(live_points, dim, resample_function=uniform_proposal)

    if args.plot:
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

        #plt.figure()
        #plt.scatter(prior_mass[:len(likelihood_worst)], likelihood_worst, s=0.1)
        #plt.xlabel('log(X)')
        #plt.ylabel('Worst Likelihood')

        plt.figure()
        plt.scatter(np.arange(len(t_resample)),t_resample, s=0.5)
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('Resampling time')

        plt.figure()
        plt.scatter(np.arange(len(prior_shrink)), prior_shrink, s=0.5)
        plt.xlabel('Iterations')
        plt.ylabel('Shrinking')

        plt.figure()
        plt.hist(prior_shrink, bins=10)
        plt.show()

    end = time.time()
    print('============ SUMMARY ============')
    print('Evidence = {0:.5f}'.format(evidence))
    print('Last area value = {0:.5f}'.format(area_plot[-1]))
    print('Check of priors sum: ', prior_mass)
    t = end-start
    print('Total time: {0:.2f} s'.format(t))
    print('=================================')
    plt.show()

