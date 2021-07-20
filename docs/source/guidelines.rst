Here there is a fast overview of the theory behind the algorithm, the challanges that I have encountered during the implementation and how I faced them. It is not meant to be a sophisticated or even correct treatment, since the project's scope was a university exam (Introduction to Bayesian Probability Theory by Walter Del Pozzo, University of Pisa) and the problem at hand is very difficult. It was not required to obtain exact results. If I will help someone to get an idea of how to handle the problem, I will be happy.

Brief Introduction
------------------

Nested Sampling is an algorithm aimed to evaluate the evidence in the Bayes' Theorem

.. math::
    P(H|DI)=\frac{P(H|I)P(D|HI)}{P(D|I)}=\frac{P(H|I)P(D|HI)}{\sum_i^{ }P(H_i|I)P(D|H_iI)}

where H is the hypothesis that explain some phenomenon, D are the data registered and I is the background information. Keep in mind that usually we can not compute directly the sum over all the possibile hypothesis that explain our phenomenon because they aren't mutually exclusive (and we don't have the power to enumerate all of them!). Using the usual names that can be found in the literature

.. math::
    Posterior=\frac{ \Prior \times Likelihood}{Evidence}

Evidence is one of the most important quantites of the Bayes Probability Theory since it enters in the computattion of the so-called Odds Ratio between two competing hypothesis H1 and H2.

.. math::
    \frac{P(H_1|DI)}{P(H_2|DI)}=\frac{P(H_1|I)}{P(H_2|I)}\frac{P(D|H_1I)}{P(D|H_2I)}=\Pr ior\ Odds\times Bayes\ Factor

Odds Ratio reduces to the Bayes Factor when we do not have any prior information that can favors one hypothesis over the other (we have to be fair). Then, writing the Bayes Theorem for the set of parameters on which H1 and H2 depend, Bayes Factor becames the ratio between the evidence of the two set of parameters (note that the integrals are actually the ratio of the likelihoods by marginalization)

.. math::
    B_{12}=\frac{\int_{\Theta_1}^{ }d\vec{\theta_1}P(\vec{\theta_1}|H_1I)P(D|\vec{\theta_1}H_1I)}{\int_{\Theta_2}^{ }d\vec{\theta_2}P(\vec{\theta_2}|H_2I)P(D|\vec{\theta_2}H_2I)}

General Idea
------------

We end up with the problem of computing a multidimensional integral of this form

.. math::
    Z=\int_{ }^{ }\int_{ }^{ }...\int_{ }^{ }L(\vec{\theta})\pi(\vec{\theta})d\vec{\theta}

The main idea to image to change variable in such a way that

.. math::
    \pi(\vec{\theta})d\vec{\theta}=d\xi

where :math: '\xi' is called the prior mass and it represents the cumulative prior over a specific level of the likelihood. It is defined by

.. math::
    \xi(\lambda) = \underset{L(\vec{\theta})>\lambda} {\int \int ... \int}\pi(\vec{\theta})d\vec{\theta}

It is clear the meaning of the prior mass looking at the following one dimension uniform prior and gaussian likelihood

.. image:: images/priormass.jpg
   :width: 150pt

If we are able to find the transformation that maps the prior into the prior mass we will end up with a 1-dim integral over the interval [0,1] insted of an N-dim integral over the entire parameter space. The problem is hugely reduced in terms of computationally complexity

.. math::
    Z=\int_0^1L(\xi)d\xi \approx \sum_i^{ }L_i\Delta\xi_i

The problem is that we do not know this transformation, but the nested sampling finds it in a statistical way, reasoning on just the fact that the likelihood is a decreasing function of the prior mass. For more details, check the original paper by Skilling (the one published in 2004 or the other in 2006. The book of Sivia and Skilling, Data Analysis, has a great treatement of the subject, too.), but to get a general idea of what you have to do, consider the following image that describes in a schematic way the major steps of the algorithm

.. image:: images/algorithm.jpg
    :width: 150pt


Problems I encountered
----------------------
The main problems I encountered are of two forms (as usually!): technical problems and conceptual problems. The formers are related to my python experience in programming that is still pretty low, the latters are due to the tricky part of the algorithm: the replacing of the worst object with a new one satifying the constraint on the likelihood.
