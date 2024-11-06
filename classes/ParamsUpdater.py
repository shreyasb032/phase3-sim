import numpy as np
from scipy.special import digamma, loggamma
from scipy.optimize import minimize
from classes.TrustModels import BetaDistributionModel
from classes.PerformanceMetrics import ObservedReward


class Estimator:
    """
    Estimates the trust parameters after getting trust feedback from the human
    """

    def __init__(self):
        """
        Initializer of the Estimator class
        current_model: A trust model that needs to be updated
        """
        self.prior = None
        self.trust_feedback = []
        self.perf_history = []

    def update_model(self, trust: float, performance: int):

        """
        Function to get the updated list of trust parameters
        :param trust: the trust feedback given by the human after observing the outcome
        :param performance: the performance of the recommendation at the current trial
        """

        self.trust_feedback.append(trust)
        self.perf_history.append(performance)
        x0 = np.ones((4,), dtype=float)
        fun = lambda x: self.neg_log_likelihood(x, self.trust_feedback, self.perf_history)
        grad = lambda x: self.gradients(x, self.trust_feedback, self.perf_history)
        bnds = ((1, 200), (1, 200), (0.1, 200), (0.1, 200))

        res = minimize(fun, x0, jac=grad, method='SLSQP', bounds=bnds)
        return res.x

    @staticmethod
    def neg_log_likelihood(x, *args):
        """
        The negative log-likelihood function
        :param x: the trust params in order [alpha0, beta0, ws, wf]
        """
        trust_history, perf_history = args
        logl = 0
        alpha0, beta0, ws, wf = x

        alpha = alpha0
        beta = beta0
        for i, (t, p) in enumerate(zip(trust_history, perf_history)):
            alpha += p * ws
            beta += (1 - p) * wf
            t = max(min(t, 0.99), 0.01)
            logl += (loggamma(alpha + beta) - loggamma(alpha) - loggamma(beta) + (alpha - 1) * np.log(t) +
                     (beta - 1) * np.log(1. - t))

        return -logl

    @staticmethod
    def gradients(x, *args):
        """
        The gradient of the log-likelihood function
        """
        grads = np.zeros_like(x)
        trust_history, perf_history = args
        num_sites = len(perf_history)
        alpha0, beta0, ws, wf = x
        ns = 0
        nf = 0

        for i in range(num_sites):
            # We need to add the number of successes and failures regardless of whether feedback was queried or not
            ns += perf_history[i]
            nf += (1 - perf_history[i])

            alpha = alpha0 + ns * ws
            beta = beta0 + nf * wf

            digamma_both = digamma(alpha + beta)
            digamma_alpha = digamma(alpha)
            digamma_beta = digamma(beta)

            delta_alpha = digamma_both - digamma_alpha + np.log(max(trust_history[i], 0.01))
            delta_beta = digamma_both - digamma_beta + np.log(max(1 - trust_history[i], 0.01))

            grads[0] += delta_alpha
            grads[1] += delta_beta
            grads[2] += ns * delta_alpha
            grads[3] += nf * delta_beta

        return -grads
