'''Functions to be used for the nested sampling algorithm (Skilling 2004)'''
import numpy as np
import matplotlib.pyplot as plt

def log_likelihood(x, dim, init, boundary=5):
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
    boundary : init, optional
        Boundary of the parameter space. The default is 5

    Returns
    --------
    Likelihood : list or float
        Likelihood values or singole likelihood value
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
    '''Return a uniform prior for each value of the N-dimension vector x.
    It is set in such a way that the integral of the product with the
    likelihood over the parameter space is 1.

    Parameters
    -----------
    x : numpy.array
        A random N dimensional vector.
    dim : int
        Dimension of the parameter space.
    boundary : init, optional
        Boundary of the parameter space. The default is 5

    Retruns
    -------
    prior : list
        List of prior values.
    '''

    prior = []
    mod = np.sqrt(dim) * boundary
    for v in x:
        if np.sqrt((v*v).sum()) > mod:
            prior.append(-np.inf)
        else:
            prior.append(-dim*np.log(2*boundary))

    return prior


def autocorrelation(x, max_lag, bootstrap=False):
    '''Simple implementation for the autocorrelation function.

    Parameters
    ----------
    x : numpy array or list
        Input array for the autocorrelation function
    max_lag : int
        Maximum lag to be used for the AC

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
