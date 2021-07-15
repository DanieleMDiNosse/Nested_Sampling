'''Functions to be used for the nested sampling algorithm (Skilling 2004)'''
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools as it

def log_likelihood(x, dim, init):
    ''' Return the logarithm of a N-dimensional gaussian likelihood.
    It is set in such a way that the integral of the product with the
    prior over the parameter space is 1.

    Parameters
    ----------
    x : numpy.array
        If init is set to True, x should be a MxN matrix whose M rows (number of points)
        are random N-dimensional vectors. In this case it is used to initialize the
        likelihood of the live points. If init is set to False, x should be just a
        random N dimensional vector.
    dim : int
        Dimension of the parameter space.
    init: bool
        You can choose to use the funcion to initialize the likelihood of the live
        points (True) or to generate just a new likelihood value (False)

    Returns
    --------
    Likelihood : list or float
        Likelihood values or singole likelihood value
    '''

    likelihood = []

    if init:
        for v in x:
            L = - 0.5*dim*np.log(2*np.pi) - 0.5*v.T.dot(v)
            likelihood.append(L)
    else:
        L = - 0.5*dim*np.log(2*np.pi) - 0.5*x.T.dot(x)
        likelihood.append(L)

    return likelihood


def autocorrelation(x, max_lag, bootstrap=False):
    '''Simple implementation for the autocorrelation function.

    Parameters
    ----------
    x : numpy array or list
        Input array for the autocorrelation function
    max_lag : int
        Maximum lag to be used for the AC
    bootstrap : bool
        Compute the bootstrap test

    Returns
    --------
    auto_corr : numpy array
        Autocorrelation function'''

    x = np.array(x)
    x_mean = np.mean(x)
    auto_corr = []
    for d in range(max_lag):
        ac = 0
        for i in range(len(x)-d):
            ac += (x[i] - x_mean) * (x[i+d] - x_mean)
        ac = ac / np.sqrt(np.sum((x - x_mean)**2) * np.sum((x - x_mean)**2))
        auto_corr.append(ac)
    auto_corr = np.array(auto_corr)
    plt.figure()
    plt.plot(auto_corr, 'k--', linewidth=0.4, alpha=0.5)
    plt.scatter(np.arange(len(auto_corr)), auto_corr, s=5, color='black')
    plt.grid()
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')


    if bootstrap:
        auto_corr_bootstrap = []
        for i in range(200):
            xs = x
            np.random.shuffle(xs)
            xs_mean = np.mean(xs)
            auto_corr_bootstrap_i = []
            for d in range(max_lag):
                ac = 0
                for i in range(len(x)-d):
                    ac += (x[i] - x_mean) * (x[i+d] - x_mean)
                ac = ac / np.sqrt(np.sum((x - x_mean)**2) * np.sum((x - x_mean)**2))
                auto_corr_bootstrap_i.append(ac)
            auto_corr_bootstrap.append(auto_corr_bootstrap_i)
        meanac = np.mean(np.array(auto_corr_bootstrap), axis=0)
        stdac = np.std(np.array(auto_corr_bootstrap), axis=0)
        plt.plot(meanac - 2*stdac, 'black', lw=0.5)
        plt.plot(meanac, 'black', lw=0.5)
        plt.plot(meanac + 2*stdac, 'black', lw=0.5)
        plt.fill_between(np.arange(0, max_lag), meanac + 2*stdac, meanac - 2*stdac, color='black', alpha=0.4)
    plt.show()

    return auto_corr

def proposal(x, dim, logLmin, boundary_point, boundary, std, distribution):
    ''' Sample a new object from the prior subject to the constrain L(x_new) > Lworst_old

    Parameters
    ----------
    x : numpy array
        Array (parameter, prior, likelihood) corresponding to the worst likelihood
    dim : int
        Dimension of the parameter space.
    logLmin : float64
        Worst likelihood, i.e. The third element of x
    std : float
        Limits of the uniform distribution proposal or standard deviation of the normal/anglit
        distribution. The name comes from the fact that it correspondsto the mean of standard
        deviations of the points along the axis of the parameter space
    distribution : string
        Choose the distribution from which the new object should be sampled.
        Available options are 'uniform', 'normal'

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
    c = 0

    k_u = 1
    loop = it.cycle(np.arange(1.5,7.5,1.5))
    k_n = np.log(dim+5)
    k_a = 1
    while True:
        new_line = np.zeros(dim+1, dtype=np.float64)
        #k_n = next(loop)
        for i in range(len(new_line[:dim])):

            if distribution == 'uniform':
                new_line[:dim][i] = boundary_point[i] + np.random.uniform(-k_u*std, k_u*std)
                while np.abs(new_line[:dim][i]) > boundary:
                    new_line[:dim][i] = boundary_point[i] + np.random.uniform(-k_u*std, k_u*std)

            if distribution == 'normal':
                new_line[:dim][i] = np.random.normal(boundary_point[i], k_n*std)
                while np.abs(new_line[:dim][i]) > boundary:
                    new_line[:dim][i] = np.random.normal(boundary_point[i], k_n*std)

        new_line[dim] = log_likelihood(new_line[:dim], dim, init=False)[0]

        if new_line[dim] < logLmin:
            rejected += 1
        if new_line[dim] > logLmin:
            n += 1
            accepted += 1
            boundary_point[:dim] = new_line[:dim]
            if n > 10:
                end = time.time()
                t = end - start
                break
        if distribution == 'uniform':
            if accepted != 0 and rejected != 0:
                if accepted > rejected: std *= np.exp(1.0/accepted)
                if accepted < rejected: std /= np.exp(1.0/rejected)

    return new_line, t, accepted, rejected
