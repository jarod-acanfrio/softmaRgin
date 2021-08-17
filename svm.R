hadamard <- function(scalar_vec, m) {

  assertthat::are_equal(nrow(scalar_vec), nrow(m))
  out_matrix <- matrix(nrow = nrow(m), ncol = ncol(m))
  for (i in 1:ncol(out_matrix)) { out_matrix[,i] <- scalar_vec * m[,i] }
  return(out_matrix)
}

svm <- function(x,
                y,
                maxiter = 1000,
                tol = 1e-4,
                beta = 1,
                lambda = 1,
                print_status = FALSE) {

  Nx <- nrow(x)
  px <- ncol(x)

  Ny <- nrow(y)
  py <- ncol(y)

  if(Nx != Ny) { stop() }

  # Randomly initialize the weights and bias
  w <- as.matrix(rnorm(px))
  b <- as.matrix(rnorm(1))

  # Declare variables
  d <- as.matrix(rep(0, Nx)) # Dual variables
  ones <- as.matrix(rep(1, Nx)) # Unit Vector

  X <- cbind(x, ones) # Augmented data and ones
  W <- rbind(w, b) # Augmented weights and bias
  B <- diag(px) |>
    cbind(as.matrix(rep(0, px))) |>
    rbind(t(as.matrix(rep(0, px+1)))) # Quadratic Form Matrix
  A <- hadamard(y, X) # Element-wise label - feature matrix
  S <- ones - A %*% W # Margin Violations
  Q <- as.matrix(rep(1, Nx)) # Proximal Mapping
  BAtAinv <- matlib::inv(B + beta*(t(A) %*% A)) # Closed Form Update Constant

  # Set up residual histories
  pres <- 1
  dres <- 1
  hist_pres <- pres
  hist_dres <- dres

  # Algorithm body
  iter <- 0
  while(max(pres, dres) > tol & iter < maxiter) {
    iter <- iter + 1

    # Primal Variable Update
    W = BAtAinv %*% t(A) %*% (beta * (ones - S) - d)

    Sprev <- S
    Q = ones - A %*% W - 0.5 * d

    # Proximal Mapping for Dual Variable Update
    for (i in 1:Nx) {
      if(Q[i] < 0) {
        S[i] <- Q[i]
      } else if (0 <= Q[i] & Q[i] <= 1/(beta * lambda)) {
        S[i] < 0
      } else {
        S[i] <- Q[i] - 1 / (beta * lambda)
      }
    }

    d <- d + beta * (A %*% W + S - ones)

    pres <- norm(A %*% W + S - ones, type = "2")
    dres <- norm(beta * t(A) %*% (Sprev - S), type = "2")
    hist_pres <- c(hist_pres, pres)
    hist_dres <- c(hist_dres, dres)

    if(print_status) {
      cat("Iteration : ", iter, " Primal Residual : ", pres, " Dual Residual : ", dres, "\n")
    }

  }

  cat("Converged at ", iter, " iterations.\n")
  cat("Primal Residual : ", pres, "\n")
  cat("Dual Residual : ", dres, "\n")

  svm_rtrn <- list(
    iterations = iter,
    primal_history = hist_pres,
    dual_history = hist_dres,
    beta = beta,
    lambda = lambda,
    weights = as.matrix(W[-nrow(W)]),
    bias = as.matrix(W[nrow(W)])
  )
  class(svm_rtrn) <- "softmaRgin"

  return(svm_rtrn)
}
