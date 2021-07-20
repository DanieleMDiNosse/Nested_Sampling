Here there is a fast overview of the theory behind the algorithm, the challanges that I have encountered during the implementation and how I faced them. It is not meant to be a sophisticated or even correct treatment, since the project's scope was a university exam (Introduction to Bayesian Probability Theory by Walter Del Pozzo, University of Pisa) and the problem at hand is very difficult. It was not required to obtain exact results. If I will help someone to get an idea of how to handle the problem, I will be happy.

Brief Introduction
------------------

Nested Sampling is an algorithm aimed to evaluate the evidence in the Bayes' Theorem

.. math::
    P(H|DI)=\frac{P(H|I)P(D|HI)}{P(D|I)}=\frac{P(H|I)P(D|HI)}{\sum_i^{ }P(H_i|I)P(D|H_iI)}

where H is the hypothesis that explain some phenomenon, D are the data registered and I is the background information. Keep in mind that usually we can not compute directly the sum over all the possibile hypothesis that explain our phenomenon because they aren't mutually exclusive (and we don't have the power to enumerate all of them!). Using the usual names that can be found in the literature

.. math::
    Posterior=\frac{\Pr ior\times Likelihood}{Evidence}

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
