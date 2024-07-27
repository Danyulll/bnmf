models <- list(
  S_gamma_A_gamma = "
model {
    # Likelihood
    for (i in 1:m) {
        for (k in 1:n) {
            X[i, k] ~ dnorm(mu[i, k], tau[i])
            mu[i, k] <- inprod(A[i, ], S[, k])
        }
    }

    # Prior distributions for pure spectra (S) and mixing coefficients (A)
    for (j in 1:p) {
        for (k in 1:n) {
            S[j, k] ~ dgamma(alpha_s[j], beta_s[j]) T(0.0001,)
        }
    }

    for (i in 1:m) {
      for (j in 1:p) {
        A[i, j] ~ dgamma(alpha_a[j], beta_a[j]) T(0.0001,)
      }
    }


    # Hyperparameters for the Gamma distributions for each element of S
    for (j in 1:p) {

            alpha_s[j] ~ dgamma(2, E)
            beta_s[j] ~ dgamma(2, E)
            alpha_a[j] ~ dgamma(2, E)
            beta_a[j] ~ dgamma(2, E)

    }

    # Noise variances
    for (i in 1:m) {
        tau[i] ~ dgamma(2, E)
    }
}
",
S_exp_A_gamma ="
model {
    # Likelihood
    for (i in 1:m) {
        for (k in 1:n) {
            X[i, k] ~ dnorm(mu[i, k], tau[i])
            mu[i, k] <- inprod(A[i, ], S[, k])
        }
    }


    # Prior distributions for pure spectra (S) and mixing coefficients (A)
    for (j in 1:p) {
        for (k in 1:n) {
            S[j, k] ~ dexp(lambda_s[j,k]) T(0.0001,)
        }
    }

    for (i in 1:m) {
      for (j in 1:p) {
        A[i, j] ~ dgamma(alpha_a[i,j], beta_a[i,j]) T(0.0001,)
      }
    }


    # Hyperparameters for the Gamma distributions for each element of S
    for (j in 1:p) {
        for (k in 1:n) {
           lambda_s[j,k] ~ dgamma(2, E) # Assuming a gamma prior for the rate parameters of the Exponential distribution
        }
    }

      for (i in 1:m) {
          for (j in 1:p) {
            alpha_a[i,j] ~ dgamma(2, E)
            beta_a[i,j] ~ dgamma(2, E)
          }
        }


    # Noise variances
    for (i in 1:m) {
        tau[i] ~ dgamma(2, E)
    }
}
",
S_half_norm_A_gamma = "
model {
    # Likelihood
    for (i in 1:m) {
        for (k in 1:n) {
            X[i, k] ~ dnorm(mu[i, k], tau[i])
            mu[i, k] <- inprod(A[i, ], S[, k])
        }
    }

    # Prior distributions for pure spectra (S) and mixing coefficients (A)
    for (j in 1:p) {
        for (k in 1:n) {
            S[j, k] ~ dnorm(mu_s[j,k], tau_s[j,k]) T(0,)
        }
    }

    for (i in 1:m) {
        for (j in 1:p) {
            A[i, j] ~ dgamma(alpha_a[i,j], beta_a[i,j]) T(0.0001,)
        }
    }

    # Hyperparameters for the Gamma distributions for each element of S


     for (j in 1:p) {
        for (k in 1:n) {
            tau_s[j,k] ~ dgamma(2, E)
            mu_s[j,k] ~ dgamma(2, E)
        }
    }

    for (i in 1:m) {
        for (j in 1:p) {
          alpha_a[i,j] ~ dgamma(2, E)
          beta_a[i,j] ~ dgamma(2, E)
        }
    }

    # Noise variances
    for (i in 1:m) {
        tau[i] ~ dgamma(2, E)
    }
}
",
S_gamma_A_exp ="
model {
    # Likelihood
    for (i in 1:m) {
        for (k in 1:n) {
            X[i, k] ~ dnorm(mu[i, k], tau[i])
            mu[i, k] <- inprod(A[i, ], S[, k])
        }
    }

    # Prior distributions for pure spectra (S)
     for (j in 1:p) {
        for (k in 1:n) {
            S[j, k] ~ dgamma(alpha_s[j, k], beta_s[j, k]) T(0.0001,)
        }
    }

    # Prior distributions for mixing coefficients (A) using Exponential distribution
    for (i in 1:m) {
      for (j in 1:p) {
        A[i, j] ~ dexp(lambda_a[i,j]) T(0.0001,)
      }
    }


     for (j in 1:p) {
        for (k in 1:n) {
            alpha_s[j,k] ~ dgamma(2, E)
            beta_s[j,k] ~ dgamma(2, E)
        }
    }

     for (i in 1:m) {
      for (j in 1:p) {
        lambda_a[i,j] ~ dgamma(2, E)
    }}

    # Noise variances
    for (i in 1:m) {
        tau[i] ~ dgamma(2, E)
    }
}
",
S_gamma_A_half_norm = "
model {
    # Likelihood
    for (i in 1:m) {
        for (k in 1:n) {
            X[i, k] ~ dnorm(mu[i, k], tau[i])
            mu[i, k] <- inprod(A[i, ], S[, k])
        }
    }

    # Prior distributions for pure spectra (S) and mixing coefficients (A)
    for (j in 1:p) {
        for (k in 1:n) {
            S[j, k] ~ dgamma(alpha_s[j,k], beta_s[j,k]) T(0.0001,1.001)
        }
    }

    for (i in 1:m) {
        for (j in 1:p) {
            A[i, j] ~ dnorm(mu_a[i,j], tau_a[i,j]) T(0,)
        }
    }

    # Hyperparameters for the Gamma distributions for each element of S

     for (i in 1:m) {
        for (j in 1:p) {
            tau_a[i,j] ~ dgamma(2, E)
            mu_a[i,j] ~ dgamma(2, E)
        }
    }

     for (j in 1:p) {
        for (k in 1:n) {
          alpha_s[j,k] ~ dgamma(2, E)
          beta_s[j,k] ~ dgamma(2, E)
        }
    }


    # Noise variances
    for (i in 1:m) {
        tau[i] ~ dgamma(2, E)
    }
}
",
S_gamma_A_gamma_t_likelihood = "
model {
  # Likelihood
  for (i in 1:m) {
    for (k in 1:n) {
      X[i, k] ~ dt(mu[i, k], tau[i], nu)
      mu[i, k] <- inprod(A[i, ], S[, k])
    }
  }

  # Prior distributions for pure spectra (S) and mixing coefficients (A)
  for (j in 1:p) {
    for (k in 1:n) {
      S[j, k] ~ dgamma(alpha_s[j,k], beta_s[j,k]) T(0.0001,1.001)
    }
  }

  for (i in 1:m) {
    for (j in 1:p) {
      A[i, j] ~ dgamma(alpha_a[i,j], beta_a[i,j]) T(0.0001,1.001)
    }
  }


  # Hyperparameters for the Gamma distributions for each element of S
  for (i in 1:m) {
    for (j in 1:p){
      alpha_a[i,j] ~ dgamma(2, E)
      beta_a[i,j] ~ dgamma(2, E)
    }
  }

  for (j in 1:p){
    for (k in 1:n){
      alpha_s[j,k] ~ dgamma(2, E)
      beta_s[j,k] ~ dgamma(2, E)
    }
  }

  # Noise variances
  for (i in 1:m) {
    tau[i] ~ dgamma(2, E)
  }
}
"
)

#' Fit a classic Bayesian non-negative matrix factorization with or without the iid assumption
#'
#' @param X a \eqn{m\times n} data matrix you wish to decompose.
#' @param p the number of components to search for.
#' @param A a \eqn{m\times p} matrix initialization for the mixing matrix \eqn{\mathbf{A}}.
#' @param S a \eqn{p \times n} matrix initialization for the pure component matrix \eqn{\mathbf{S}}.
#' @param A_dist a string indicating the distribution for A (one of "Gamma", "Exponential", or "Normal").
#' @param S_dist a string indicating the distribution for A (one of "Gamma", "Exponential", or "Normal").
#' @param likelihood a string indicating the distribution for the likelihood of the data (either "Normal" or "T").
#' @param params a list of the hyperparameters for the distributions of \eqn{\mathbf{A}} and \eqn{\mathbf{S}}.
#' @param n.chains an integer for the number of Markov chains you wish to generate.
#' @param adapt an integer for the number of adaptation iterations you wish the Gibbs sampler to undergo.
#' @param burnin an integer for the number of burn-in iterations you wish the Gibbs sampler to undergo.
#' @param sample an integer indicating the number of samples you wish to generate.
#' @param thin an integer indicating the thinning interval for the Markov chain.
#' @param method a string indicating the method to run the Gibbs sampler with (see \link[runjags]{run.jags} for more information.)
#' @param seed seeds for reproducibility of Markov chains. Must have as many seeds as there are chains.
bnmf <- function(X,
                 p,
                 A = NA,
                 S = NA,
                 A_dist = "Gamma",
                 S_dist = "Gamma",
                 likelihood = "Normal",
                 params = list(),
                 n.chains = 4,
                 adapt = 1000,
                 burnin = 5000,
                 sample = 5000,
                 thin = 1,
                 method = "simple",
                 seed = NA) {
  if(is.data.frame(X))
    X <- data.matrix(X)

  m <- nrow(X)
  n <- ncol(X)

  # Argument checking
  if (!is.integer(p))
    stop("p must be an integer.")

  if (!is.null(A)) {
    if (nrow(A) != m)
      stop("Rows of A and X do not match.")
    if (ncol(A) != p)
      stop("Columns of A and p do not match.")
  }
  if (!is.null(S)) {
    if (nrow(S) != p)
      stop("Rows of S and p do not match.")
    if (ncol(S) != n)
      stop("Columns of S and X do not match")
  }

  if (!(A_dist %in% c("Gamma", "Normal", "Exponential")))
    stop("A distribution not currently supported.")

  if (!(S_dist %in% c("Gamma", "Normal", "Exponential")))
    stop("S distribution not currently supported.")

  if (!(likelihood %in% c("Normal", "T")))
    stop("Likelihood not currently supported.")
  if (!is.integer(n.chains))
    stop("chain must be an integer.")

  if (any(is.numeric(params)))
    stop("Incorrect type in params")

  if (A_dist == "Gamma" &&
      sum(c("alpha_a", "beta_a") %in% names(params)) != 2)
    stop("Incorret params or param names for A_dist")
  if (S_dist == "Gamma" &&
      sum(c("alpha_s", "beta_s") %in% names(params)) != 2)
    stop("Incorret params or param names for S_dist")

  if (A_dist == "Exponential" && !("lambda_a" %in% names(params)))
    stop("Incorret params or param names for A_dist")
  if (S_dist == "Exponential" && !("lambda_s" %in% names(params)))
    stop("Incorret params or param names for S_dist")

  if (A_dist == "Normal" &&
      sum(c("mu_a", "tau_a") %in% names(params)) != 2)
    stop("Incorret params or param names for A_dist")
  if (S_dist == "Normal" &&
      sum(c("mu_s", "tau_s") %in% names(params)) != 2)
    stop("Incorret params or param names for S_dist")

  if(!is.na(seed) && length(seed) != n.chains)
    stop("Incorrect number of seeds provided.")

  if (is.na(A))
    A <- matrix(runif(m * p), nrow = m, ncol = p)
  if (is.na(S))
    S <- matrix(runif(n * p), nrow = p, ncol = n)


  data.list <-  c(list(
    X =  X,
    m = m,
    n = n,
    p = p,
    E = 1 * 10 ^ -3
  ), params)

  # List of RNGs
  rng_list <- c("base::Wichmann-Hill", "base::Marsaglia-Multicarry", "base::Super-Duper", "base::Mersenne-Twister")

  # Generate the inits for the specified number of chains
  generate_inits <- function(num_chains, rng_list, seeds) {
    inits <- list()
    for (i in 1:num_chains) {
      rng_name <- rng_list[((i - 1) %% length(rng_list)) + 1]
      inits[[i]] <- runjags::dump.format(list(
        A = A,
        S = S,
        .RNG.name = rng_name,
        .RNG.seed = seeds[i]
      ))
    }
    return(inits)
  }

  inits1 <-
    runjags::dump.format(list(
      A = A,
      S = S,
      .RNG.name = "base::Wichmann-Hill",
      .RNG.seed = seed
    ))
  inits2 <-
    runjags::dump.format(list(
      A = A,
      S = S,
      .RNG.name = "base::Marsaglia-Multicarry",
      .RNG.seed = seed
    ))
  inits3 <-
    runjags::dump.format(list(
      A = A,
      S = S,
      .RNG.name = "base::Super-Duper",
      .RNG.seed = seed
    ))
  inits4 <-
    runjags::dump.format(list(
      A = A,
      S = S,
      .RNG.name = "base::Mersenne-Twister",
      .RNG.seed = seed
    ))

  model <- switch(
    paste(A_dist, S_dist, sep = "_"),
    "Gamma_Gamma" = models$S_gamma_A_gamma,
    "Gamma_Exponential" = models$S_exp_A_gamma,
    "Exponential_Gamma" = models$S_gamma_A_exp,
    "Gamma_Normal" = models$S_half_norm_A_gamma,
    "Normal_Gamma" = models$S_gamma_A_half_norm
  )

  posterior <- runjags::run.jags(
    model = model,
    data = data.list,
    monitor = c("A","S"),
    inits = c(inits1, inits2, inits3, inits4),
    n.chains = n.chains,
    adapt = adapt,
    burnin = burnin,
    sample = sample,
    thin = thin,
    method = method
  )

  posterior
}
