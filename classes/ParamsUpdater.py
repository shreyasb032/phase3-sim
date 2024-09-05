import numpy as np
from numpy.linalg import norm
from scipy.special import digamma
from classes.TrustModels import BetaDistributionModel
from classes.PerformanceMetrics import ObservedReward


class Estimator:
    """
    Estimates the trust parameters after getting trust feedback from the human
    """

    def __init__(self, current_model: BetaDistributionModel,
                 max_iterations=10000, step_size=0.01, error_tol=0.01,
                 num_sites=5):
        """
        Initializer of the Estimator class
        :param max_iterations: maximum number of iterations of gradient descent to run before stopping
        :param step_size: the step_size for the gradient descent algorithm
        :param error_tol: the error below which we stop the gradient descent algorithm
        :param num_sites: the number of sites in the mission
        """

        self.prior = None
        self.MAX_ITER = max_iterations
        self.step_size = step_size
        self.define_prior()
        self.error_tol = error_tol
        self.N = num_sites

        # Initial guesses for the trust params
        self.gp_list = {0.0: [2., 98., 20., 30.],
                        0.1: [10., 90., 20., 30.],
                        0.2: [20., 80., 20., 30.],
                        0.3: [30., 70., 20., 30.],
                        0.4: [40., 60., 20., 30.],
                        0.5: [50., 50., 20., 30.],
                        0.6: [60., 40., 20., 30.],
                        0.7: [70., 30., 20., 30.],
                        0.8: [80., 20., 20., 30.],
                        0.9: [90., 10., 20., 30.],
                        1.0: [98., 2., 20., 30.]}

        # self.feedback = np.zeros((self.N+1,), dtype=float)
        self.feedback = []
        self.keys = ['alpha0', 'beta0', 'vs', 'vf']
        self.current_model = current_model

    def define_prior(self):
        """
        Helper function to define the prior over the trust parameters
        """

        prior = {"AlphaEdges": np.array([0, 28, 56, 84, 112, 140]),
                 "AlphaValues": np.array([0.2051, 0.1538, 0.07692, 0.2308, 0.3333]),
                 "BetaEdges": np.array([0, 29, 58, 87, 116, 145]),
                 "BetaValues": np.array([0.1269, 0.2335, 0.3063, 0.1808, 0.1525]),
                 "wsEdges": np.array([0, 14, 28, 42, 56, 70]),
                 "wsValues": np.array([0.5897, 0.1795, 0.1032, 0.07625, 0.05128]),
                 "wfEdges": np.array([0, 28, 56, 84, 112, 140]),
                 "wfValues": np.array([0.5641, 0.1026, 0.05128, 0.0641, 0.2179])}

        self.prior = prior

    def get_initial_guess(self, feedback) -> BetaDistributionModel:
        """
        Get a good initial guess to start the gradient descent algorithm.
        The guess is chosen from a list to best estimate the initial value of trust given by the human
        :param feedback: the initial trust feedback given by the human
        """

        t = round(feedback * 10) / 10
        guess_params = dict(zip(self.keys, self.gp_list[t].copy()))
        performance_metric = ObservedReward()
        trust_model = BetaDistributionModel(guess_params, performance_metric)
        self.current_model = trust_model

        return trust_model

    def update_model(self, trust_feedback: float):

        """
        Function to get the updated list of trust parameters
        :param trust_feedback: the trust feedback given by the human after observing the outcome
        # :param site_idx: the index of the site in which search was just completed
        """

        t = trust_feedback
        # self.feedback[site_idx] = t
        self.feedback.append(t)

        factor = self.step_size
        lr = np.array([factor, factor, factor / self.N, factor / self.N])

        guess_params = np.array(list(self.current_model.parameters.values()))
        gradients_except_prior = self.get_grads(guess_params)
        num_iters = 0

        while norm(gradients_except_prior) > self.error_tol and num_iters < self.MAX_ITER:
            num_iters += 1
            gradients_except_prior = self.get_grads(guess_params)
            guess_params += lr * gradients_except_prior
            guess_params[guess_params <= 0.1] = 0.1  # To make sure the digamma function behaves well

        new_params = dict(zip(self.keys, guess_params))
        self.current_model.update_parameters(new_params)

        return self.current_model

    def get_grads(self, params):
        """
        Returns the gradients of the log-likelihood function using a digamma approximation
        :param params: the trust parameters at which to evaluate the gradients
        # :param current_site: the index of the current search site
        """

        grads = np.zeros((4,))
        alpha_0 = params[0]
        beta_0 = params[1]
        ws = params[2]
        wf = params[3]

        ns = 0
        nf = 0

        for i in range(len(self.feedback)):

            # We need to add the number of successes and failures regardless of whether feedback was queried or not
            ns += self.current_model.performance_history[i]
            nf += (1 - self.current_model.performance_history[i])

            # If feedback was queried here, compute the gradients
            alpha = alpha_0 + ns * ws
            beta = beta_0 + nf * wf

            digamma_both = digamma(alpha + beta)
            digamma_alpha = digamma(alpha)
            digamma_beta = digamma(beta)

            delta_alpha = digamma_both - digamma_alpha + np.log(max(self.feedback[i], 0.01))
            delta_beta = digamma_both - digamma_beta + np.log(max(1 - self.feedback[i], 0.01))

            grads[0] += delta_alpha
            grads[1] += delta_beta
            grads[2] += ns * delta_alpha
            grads[3] += nf * delta_beta

        return grads
