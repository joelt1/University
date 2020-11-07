"""
Original strategy using HMM
Authors: Joel Thomas, Liam Horrocks, Victoria Zhang
"""
import numpy as np
import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS

# Training period
ROLLING_WINDOW = 6
TRADE_DAYS_YEAR = 252
NUM_YEARS = 3


def random_normalized(d1, d2):
    """
    # Generates a random normalised matrix with dim(d1, d2)
    :param d1: Size of dimension 1 of matrix
    :param d2: Size of dimension 2 of matrix
    :return: X / X.sum(axis=1, keepdims=True): Randomly normalised matrix with given dimensions
    """
    X = np.random.random((d1, d2))
    return X / X.sum(axis=1, keepdims=True)


class MVN:
    """
    Multivariate Normal Gaussian PDF
    """
    def __init__(self, mu, sigma):
        """
        Constructor for this class
        :param mu: Mean vector of multivariate normal distribution
        :param sigma: Covariance matrix of multivariate normal distribution
        """
        self.mu = mu
        self.sigma = sigma

    def pdf(self, X):
        """
        Calculates PDF given distribution parameters
        :param X: Observation sequence
        :return: np.array(ps): array of probabilities for each x in X
        """
        ps = []
        for x in X:
            expression_1 = 1 / (((2 * np.pi)**(len(self.mu)/2)) * (np.linalg.det(self.sigma)**(1/2)))
            expression_2 = (-1/2) * ((x - self.mu).T.dot(np.linalg.inv(self.sigma))).dot((x - self.mu))
            ps.append(float(expression_1 * np.exp(expression_2)))
        return np.array(ps)


class HMM:
    """
    Continuous-observation HMM with scaling and allowing for only one observation sequence
    """
    def __init__(self, M, K):
        """
        Constructor for the class
        :param M: Total number of hidden states
        :param K: Total number of Gaussians in Gaussian Mixture Model (GMM)
        """
        self.M = M  # number of hidden states
        self.K = K  # number of Gaussians
        self.pi = None  # Initial state distribution
        self.A = None  # State transition matrix
        self.R = None  # Mixture proportions (responsibilities, probabilities of all K Gaussian)
        self.mu = None  # Means of all K Gaussians, dim = M*K*D
        self.Sigma = None  # Covariance matrices for all K Gaussians, dim = M*K*D*D

    def fit(self, X, D=1, max_iter=10, eps=1e-1):
        """
        Uses Baum-Welch together with Forward and Backward algorithms to fit HMM parameters (pi, A, B) on to training
        data
        :param X: Data to fit model on to
        :param D: Dimension of X e.g. (open, high, low, close) --> 4
        :param max_iter: Maximum number of training iterations
        :param eps: Epsilon, used for smoothing
        """
        # Assume X (observation sequence) is organized (T, D)
        T = len(X)

        # Initialise all parameters
        self.pi = np.ones(self.M) / self.M
        self.A = random_normalized(self.M, self.M)
        self.R = np.ones((self.M, self.K)) / self.K
        self.mu = np.zeros((self.M, self.K, D))
        for i in range(self.M):
            for k in range(self.K):
                random_index = np.random.choice(T)
                self.mu[i, k] = X[random_index]
        self.Sigma = np.ones((self.M, self.K, D, D))

        costs = []
        # E-STEP FOR EM OPTIMISATION ALGORITHM, estimate B, alpha, beta, gamma
        for iteration in range(max_iter):
            # Scale to solve underflow problem
            scale = np.zeros(T)

            # Calculate B so can lookup when updating alpha and beta
            B = np.zeros((self.M, T))
            component = np.zeros((self.M, self.K, T))
            for j in range(self.M):
                for k in range(self.K):
                    # Probability of observing jth observation at time t given symbol k
                    p = self.R[j, k] * MVN(self.mu[j, k], self.Sigma[j, k]).pdf(X)
                    component[j, k, :] = p
                    # Probability = sum of individual mixture components
                    B[j, :] += p

            # FORWARD ALGORITHM
            alpha = np.zeros((T, self.M))
            # Initialisation step
            alpha[0] = self.pi * B[:, 0]
            scale[0] = alpha[0].sum()
            alpha[0] /= scale[0]
            # Induction step
            for t in range(1, T):
                alpha_t_prime = alpha[t - 1].dot(self.A) * B[:, t]
                scale[t] = alpha_t_prime.sum()
                alpha[t] = alpha_t_prime / scale[t]
            # Termination step
            logP = np.log(scale).sum()

            # BACKWARD ALGORITHM
            beta = np.zeros((T, self.M))
            # Initialisation step
            beta[-1] = 1
            # Induction step
            for t in range(T - 2, -1, -1):
                for i in range(self.M):
                    factor = 0
                    for j in range(self.M):
                        factor += self.A[i, j] * B[j, t + 1] * beta[t + 1, j]
                    beta[t, i] = factor / scale[t + 1]

            # Update for Gaussians
            gamma = np.zeros((T, self.M, self.K))
            for t in range(T):
                alpha_beta = alpha[t, :].dot(beta[t, :])
                for j in range(self.M):
                    factor = alpha[t, j] * beta[t, j] / alpha_beta
                    for k in range(self.K):
                        gamma[t, j, k] = factor * component[j, k, t] / B[j, t]

            # Cost = log-likelihood for all observation sequences
            costs.append(logP)

            # M-STEP FOR EM OPTIMISATION ALGORITHM, re-estimate pi, A, R, mu, Sigma
            for i in range(self.M):
                self.pi[i] = alpha[0, i] * beta[0, i]

            # Numerator for A
            A_num = np.zeros((self.M, self.M))
            # Denominator for A
            A_den = np.zeros((self.M, 1))

            for i in range(self.M):
                for t in range(T - 1):
                    A_den[i] += alpha[t, i] * beta[t, i]
                    for j in range(self.M):
                        A_num[i, j] += alpha[t, i] * beta[t + 1, j] * self.A[i, j] * B[j, t + 1] / scale[t + 1]
            # Update A
            self.A = A_num / A_den

            # Update individual mixture components
            R_num_n = np.zeros((self.M, self.K))
            R_den_n = np.zeros(self.M)
            for j in range(self.M):
                for k in range(self.K):
                    for t in range(T):
                        R_num_n[j, k] += gamma[t, j, k]
                        R_den_n[j] += gamma[t, j, k]
            R_num = R_num_n
            R_den = R_den_n

            mu_num_n = np.zeros((self.M, self.K, D))
            Sigma_num_n = np.zeros((self.M, self.K, D))
            for j in range(self.M):
                for k in range(self.K):
                    for t in range(T):
                        # Update means
                        mu_num_n[j, k] += gamma[t, j, k] * X[t]

                        # Update covariances
                        Sigma_num_n[j, k] += gamma[t, j, k] * (X[t] - self.mu[j, k]) ** 2
            mu_num = mu_num_n
            Sigma_num = Sigma_num_n

            # Update R, mu, sigma
            for j in range(self.M):
                for k in range(self.K):
                    self.R[j, k] = R_num[j, k] / R_den[j]
                    self.mu[j, k] = mu_num[j, k] / R_num[j, k]
                    self.Sigma[j, k] = Sigma_num[j, k] / R_num[j, k] + np.ones(D) * eps

        # print(f"Final pi: {self.pi}")
        # print(f"Final A: {self.A}")
        # print(f"Final mu: {self.mu}")
        # print(f"Final Sigma: {self.Sigma}")
        # print(f"Final R: {self.R}")

        # plt.figure(figsize=(10, 6))
        # plt.plot(costs)
        # plt.tight_layout()
        # plt.show()

    def log_likelihood(self, X):
        """
        Returns log(P(X|model))
        :param X: Observation sequence
        :return: np.log(scale).sum(): Probability of an observation sequence given parameters
        """
        # FORWARD ALGORITHM
        T = len(X)
        scale = np.zeros(T)
        B = np.zeros((self.M, T))
        for j in range(self.M):
            for k in range(self.K):
                p = self.R[j, k] * MVN(self.mu[j, k], self.Sigma[j, k]).pdf(X)
                B[j, :] += p

        alpha = np.zeros((T, self.M))
        # Initialisation step
        alpha[0] = self.pi * B[:, 0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        # Induction step
        for t in range(1, T):
            alpha_t_prime = alpha[t - 1].dot(self.A) * B[:, t]
            scale[t] = alpha_t_prime.sum()
            alpha[t] = alpha_t_prime / scale[t]
        # Termination step
        return np.log(scale).sum()

    def predict_similar_likelihood(self, X, T, likelihood, epsilon=10):
        """
        Uses current log likelihood on trained window to compare and retrieve similar likelihoods based on past windows
        :param X: Observation sequence
        :param T: Length of observation sequence
        :param likelihood: Likelihood of observation sequence given HMM parameters
        :param epsilon: Threshold for similar likelihoods
        :return: prediction: Predicted value at end of next period
        """
        # List of all similar past likelihoods
        likelihoods = []
        # List of observation sequences for which similar likelihoods
        historical_windows = []
        # List of time periods to move back to to obtain similar likelihoods on historical window
        ts = []
        for t in range(1, T-ROLLING_WINDOW):
            X_past = X[T-ROLLING_WINDOW-t:T-t]
            likelihood_past = self.log_likelihood(X_past)
            difference = likelihood_past - likelihood
            if abs(difference) < epsilon:
                likelihoods.append(likelihood_past)
                historical_windows.append(X_past)
                ts.append(t)
        # If no similar likelihoods --> no prediction
        try:
            highest_similar_likelihood_data = historical_windows[likelihoods.index(max(likelihoods))]
            highest_similar_likelihood = likelihoods[likelihoods.index(max(likelihoods))]
            highest_similar_likelihood_t = ts[likelihoods.index(max(likelihoods))]
        except ValueError:
            return None
        # Prediction formula from appendix
        prediction = X[-1] + (X[T-highest_similar_likelihood_t] - highest_similar_likelihood_data[-1]) * \
            np.sign(likelihood - highest_similar_likelihood)
        return prediction


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    # Schedule predict_and_open_trades to run every Friday 1 hour before market close
    algo.schedule_function(
        predict_and_open_trades,
        algo.date_rules.week_end(0),
        algo.time_rules.market_close(hours=1),
    )

    # SID for SPDR S&P 500 ETF
    context.spy = sid(8554)


def close_trade(context):
    """
    Close an open trade
    """
    spy = context.spy
    order_size = context.portfolio.positions[spy].amount
    if order_size != 0:
        order_target_percent(spy, 0)


def predict_and_open_trades(context, data):
    """
    Original strategy from report
    """
    spy = context.spy
    price = data.history(spy, fields="price", bar_count=TRADE_DAYS_YEAR*NUM_YEARS, frequency="1d")
    # Resample prices to weekly
    price = price.resample("1W").last()
    X = np.array(price, dtype=float)
    T = len(X)

    # Initialise HMM model with 2 hidden states and 2 Gaussians
    hmm_model = HMM(2, 2)
    X_train = X[-ROLLING_WINDOW:]
    hmm_model.fit(X_train)
    likelihood = hmm_model.log_likelihood(X_train)
    prediction = hmm_model.predict_similar_likelihood(X, T, likelihood)
    # Record predictions for later
    record("Prediction", prediction, "Price", price[-1])
    if prediction:
        # If prediction greater than last price, open long position with entire portfolio value
        if prediction > price.iloc[-1]:
            order_target_percent(spy, 1.0)
        # Keep open positions open unless next period's prediction is lower than last price
        elif prediction < price.iloc[-1]:
            close_trade(context)
