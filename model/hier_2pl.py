import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp
import json
from numpyro.distributions.util import vec_to_tril_matrix
from jax import random

class Hier2PL:
    def __init__(self):
        with open('jax_posteriordb/data/sat.json', 'r') as file:
            data = json.load(file)
        self.y = np.array(data['y'])
        self.I = data['I']
        self.J = data['J']
        self.N = data['N']
        self.ii = np.array(data['ii'])
        self.jj = np.array(data['jj'])
        self.n = 2 * self.I + self.J + 5
        self.test_N = self.N // 5
        self.valid_N = self.N // 5
        self.train_N = int(self.N - self.valid_N - self.test_N)
        perm = np.random.RandomState(0).permutation(self.N).astype(int)
        self.y_train = self.y[perm[:self.train_N]]
        self.ii_train = self.ii[perm[:self.train_N]]
        self.jj_train = self.jj[perm[:self.train_N]]

        self.y_valid = self.y[perm[self.train_N:self.train_N + self.valid_N]]
        self.ii_valid = self.ii[perm[self.train_N:self.train_N + self.valid_N]]
        self.jj_valid = self.jj[perm[self.train_N:self.train_N + self.valid_N]]

        self.y_test = self.y[perm[self.train_N + self.valid_N:]]
        self.ii_test = self.ii[perm[self.train_N + self.valid_N:]]
        self.jj_test = self.jj[perm[self.train_N + self.valid_N:]]

    def log_prior(self, theta):
        x, ab, mu, tau, L = self.theta2par(theta)
        log_p_L = jnp.sum(dist.Normal().log_prob(L))
        exp_tau = jnp.exp(tau)
        log_p_tau = jnp.sum(dist.Exponential(0.1).log_prob(exp_tau)) + jnp.sum(tau)
        tril = vec_to_tril_matrix(L, diagonal=-1) + jnp.identity(2)
        s = jnp.matmul(jnp.diag(exp_tau), tril)
        log_p_mu = dist.Normal(0,1).log_prob(mu[0]) + dist.Normal(0, 5).log_prob(mu[1])
        log_p_ab = jnp.sum(dist.Normal().log_prob(ab))
        log_p_x = jnp.sum(dist.MultivariateNormal(mu, scale_tril=s).log_prob(x))
        return log_p_L + log_p_tau + log_p_x + log_p_ab + log_p_mu

    def theta2par(self, theta):
        x = theta[:self.I * 2].reshape((self.I, 2))
        ab = theta[self.I:self.I + self.J]
        mu = theta[self.I + self.J: self.I + self.J + 2]
        tau = theta[self.I + self.J + 2: self.I + self.J + 4]
        L = theta[self.I + self.J + 4: self.I + self.J + 5]
        return x, ab, mu, tau, L


    def log_likelihoods(self, theta, *args, **kwargs):
        x, ab, mu, tau, L = self.theta2par(theta)
        return dist.Bernoulli(logits=jnp.exp(x[...,0][self.ii_train]) * (ab[self.jj_train] - x[...,1][self.ii_train])).log_prob(self.y_train)

    def log_likelihood(self, theta, *args, **kwargs):
        return jnp.sum(self.log_likelihoods(theta))

    def valid_log_likelihoods(self, theta):
        x, ab, mu, tau, L = self.theta2par(theta)
        return dist.Bernoulli(logits=jnp.exp(x[...,0][self.ii_valid]) * (ab[self.jj_valid] - x[...,1][self.ii_valid])).log_prob(self.y_valid)

    def test_log_likelihoods(self, theta):
        x, ab, mu, tau, L = self.theta2par(theta)
        return dist.Bernoulli(logits=jnp.exp(x[...,0][self.ii_test]) * (ab[self.jj_test] - x[...,1][self.ii_test])).log_prob(self.y_test)


if __name__ == '__main__':
    cls = Hier2PL()
    sd = random.PRNGKey(3)
    theta = random.normal(sd, (cls.n,))
    print(cls.log_prior(theta), jnp.mean(cls.log_likelihoods(theta)), jnp.mean(cls.valid_log_likelihoods(theta)), jnp.mean(cls.test_log_likelihoods(theta)))