% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/bnmf.R
\name{bnmf}
\alias{bnmf}
\title{Fit a classic Bayesian non-negative matrix factorization with or without the iid assumption}
\usage{
bnmf(
  X,
  p,
  A = NA,
  S = NA,
  A_dist = "Gamma",
  S_dist = "Gamma",
  likelihood = "Normal",
  params = list(),
  n.chains = 4L,
  adapt = 1000,
  burnin = 5000,
  sample = 5000,
  thin = 1,
  method = "simple",
  seed = NA
)
}
\arguments{
\item{X}{a \eqn{m\times n} data matrix you wish to decompose.}

\item{p}{the number of components to search for.}

\item{A}{a \eqn{m\times p} matrix initialization for the mixing matrix \eqn{\mathbf{A}}.}

\item{S}{a \eqn{p \times n} matrix initialization for the pure component matrix \eqn{\mathbf{S}}.}

\item{A_dist}{a string indicating the distribution for A (one of "Gamma", "Exponential", or "Normal").}

\item{S_dist}{a string indicating the distribution for A (one of "Gamma", "Exponential", or "Normal").}

\item{likelihood}{a string indicating the distribution for the likelihood of the data (either "Normal" or "T").}

\item{params}{a list of the hyperparameters for the distributions of \eqn{\mathbf{A}} and \eqn{\mathbf{S}}.}

\item{n.chains}{an integer for the number of Markov chains you wish to generate.}

\item{adapt}{an integer for the number of adaptation iterations you wish the Gibbs sampler to undergo.}

\item{burnin}{an integer for the number of burn-in iterations you wish the Gibbs sampler to undergo.}

\item{sample}{an integer indicating the number of samples you wish to generate.}

\item{thin}{an integer indicating the thinning interval for the Markov chain.}

\item{method}{a string indicating the method to run the Gibbs sampler with (see \link[runjags]{run.jags} for more information.)}

\item{seed}{seeds for reproducibility of Markov chains. Must have as many seeds as there are chains.}
}
\description{
Fit a classic Bayesian non-negative matrix factorization with or without the iid assumption
}
