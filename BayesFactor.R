# Implements a k-sample test, extension from:
# Holmes, C. C., Caron, F., Griffin, J. E., and Stephens, D. A. (2015).
# Two-sample Bayesian nonparametric hypothesis testing.
# Bayesian Analysis, 10(2):297–320.
pt_d_sample_test <- function(X, Y, c = 1, max_depth = -1, qdist = qnorm, aj = function(depth) depth^2, log_BF = FALSE) {
  old_expressions <- options()$expressions
  options(expressions = max(max_depth, old_expressions))
  
  max_depth <- ifelse(max_depth < 0, max(1, floor(log2(length(X)) / 2)), max_depth)
  
  binary <- if (.is_discrete(X)) X else Y   #if TRUE, binary assigned to X, continuous assigned to Y
  continuous <- if (.is_discrete(X)) Y else X
  
  data <- cbind(scale(continuous), binary)  #combine data frame into a matrix
  X <- data[, 1]
  
  p_H0 <- .pt_marginal_likelihood(X, low = 0, up = 1, c = c, depth = 1, max_depth, qdist, aj)
  
  discrete_values <- if (length(unique(binary)) == 2) binary[1] else unique(binary)
  
  p_H1 <- max(sapply(discrete_values, function(i) {
    X1 <- data[data[, 2] == i, 1]
    X2 <- data[data[, 2] != i, 1]
    
    p_x1 <- .pt_marginal_likelihood(X1, low = 0, up = 1, c = c, depth = 1, max_depth, qdist, aj)
    p_x2 <- .pt_marginal_likelihood(X2, low = 0, up = 1, c = c, depth = 1, max_depth, qdist, aj)
    
    p_x1 + p_x2
  }))
  
  n_hypotheses <- length(discrete_values)
  bf <- p_H0 - p_H1 + log(n_hypotheses) # Bayes Factor with a Bonferroni-type correction
  
  options(expressions = old_expressions)
  
  if (log_BF)
    return(list(bf = bf, p_H0 = NULL, p_H1 = NULL))
  
  bf <- exp(bf)
  return(list(bf = bf, p_H0 = 1 - 1 / (1 + bf), p_H1 = 1 / (1 + bf)))
}


.pt_marginal_likelihood <- function(data, low, up, c, depth, max_depth, qdist, aj) {
  if (depth == max_depth) {
    return(0)
  }
  
  if (length(low) == 1) {
    n_j <- c(length(which((qdist(low) < data) & (data <= qdist((low + up) / 2)))),
             length(which((qdist((low + up) / 2) < data) & (data <= qdist(up)))))
  } else {
    n_j <- c(length(which((qdist(low[1]) < data[, 1]) & (data[, 1] <= qdist((low[1] + up[1]) / 2)) &
                            (qdist(low[2]) < data[, 2]) & (data[, 2] <= qdist((low[2] + up[2]) / 2)))),
             length(which((qdist((low[1] + up[1]) / 2) < data[, 1]) & (data[, 1] <= qdist(up[1])) &
                            (qdist(low[2]) < data[, 2]) & (data[, 2] <= qdist((low[2] + up[2]) / 2)))),
             length(which((qdist(low[1]) < data[, 1]) & (data[, 1] <= qdist((low[1] + up[1]) / 2)) &
                            (qdist((low[2] + up[2]) / 2) < data[, 2]) & (data[, 2] <= qdist(up[2])))),
             length(which((qdist((low[1] + up[1]) / 2) < data[, 1]) & (data[, 1] <= qdist(up[1])) &
                            (qdist((low[2] + up[2]) / 2) < data[, 2]) & (data[, 2] <= qdist(up[2])))))
  }
  
  if (sum(n_j) == 0) {
    return(0)
  }
  
  # a_j <- c * depth ^ 2
  a_j <- c * aj(depth)
  
  if (length(n_j) == 2) {
    logl <- lbeta(n_j[1] + a_j, n_j[2] + a_j) - lbeta(a_j, a_j)
  } else {
    logl <- .lmbeta(n_j[1] + a_j, n_j[2] + a_j, n_j[3] + a_j, n_j[4] + a_j) -
      .lmbeta(a_j, a_j, a_j, a_j)
  }
  
  if (length(low) == 1) {
    likelihoods <- c(.pt_marginal_likelihood(data, low, (low + up) / 2, c, depth + 1, max_depth, qdist, aj),
                     .pt_marginal_likelihood(data, (low + up) / 2, up, c, depth + 1, max_depth, qdist, aj))
  } else {
    likelihoods <- c(.pt_marginal_likelihood(data, low, (low + up) / 2, c, depth + 1, max_depth, qdist, aj),
                     .pt_marginal_likelihood(data, (low + up) / 2, up, c, depth + 1, max_depth, qdist, aj),
                     .pt_marginal_likelihood(data,
                                             c(low[1], (low[2] + up[2]) / 2),
                                             c((low[1] + up[1]) / 2, up[2]),
                                             c, depth + 1, max_depth, qdist, aj),
                     .pt_marginal_likelihood(data,
                                             c((low[1] + up[1]) / 2, low[2]),
                                             c(up[1], (low[2] + up[2]) / 2),
                                             c, depth + 1, max_depth, qdist, aj))
  }
  
  return(logl + sum(likelihoods))
}

.lmbeta <- function(...) {
  sum(lgamma(c(...))) - lgamma(sum(c(...)))
}

.is_discrete <- function(X) {
  all(X %in% 0:10)     #evaluates to TRUE only if every element of X lies within the specified range of 0 to 10.
}

