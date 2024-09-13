data {
  int<lower=1> I; // # items
  int<lower=1> J; // # persons
  int<lower=1> N; // # observations
  array[N] int<lower=1, upper=I> ii; // item for n
  array[N] int<lower=1, upper=J> jj; // person for n
  array[N] int<lower=0, upper=1> y; // correctness for n
}
parameters {
  vector[J] theta; // abilities
  vector[I] xi1;
  vector[I] xi2;
  vector[2] mu; // vector for alpha/beta means
  vector<lower=0>[2] tau; // vector for alpha/beta residual sds
  cholesky_factor_corr[2] L_Omega;
}
transformed parameters {
  vector[I] alpha;
  vector[I] beta;
  array[I] vector[2] xi; // alpha/beta pair vectors
  for (i in 1 : I) {
    xi[i, 1] = xi1[i];
    xi[i, 2] = xi2[i];
    alpha[i] = exp(xi[i, 1]);
    beta[i] = xi[i, 2];
  }
}
model {
  matrix[2, 2] L_Sigma;
  L_Sigma = diag_pre_multiply(tau, L_Omega);
  for (i in 1 : I) {
    target += multi_normal_cholesky_lpdf(xi[i] | mu, L_Sigma);
  }
  theta ~ normal(0, 1);
  L_Omega ~ lkj_corr_cholesky(4);
  mu[1] ~ normal(0, 1);
  tau[1] ~ exponential(.1);
  mu[2] ~ normal(0, 5);
  tau[2] ~ exponential(.1);
  y ~ bernoulli_logit(alpha[ii] .* (theta[jj] - beta[ii]));
}
generated quantities {
  corr_matrix[2] Omega;
  Omega = multiply_lower_tri_self_transpose(L_Omega);
}