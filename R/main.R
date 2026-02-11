#' L1 rotation for Generalized Bi-factor Models
#'
#' @param A Input factor loading matrix (p x q)
#' @param Bstart Optional starting value for B matrix. Defaults to A.
#'   Ignored when \code{nstart > 1}.
#' @param Phi Optional starting correlation matrix. Defaults to identity.
#' @param rho Initial penalty parameter for augmented Lagrangian method. Default 1.
#' @param t Initial step size. Default 1/1000.
#' @param maxit.ou Maximum outer iterations. Default 5000.
#' @param maxit.in Maximum inner iterations. Default 300.
#' @param orthogonal Logical; if TRUE, constrains factors to be orthogonal. Default FALSE.
#' @param tol1 Convergence tolerance for parameter changes. Default 1e-6.
#' @param tol2 Tolerance for backtracking line search. Default 1e-4.
#' @param verbose Logical; print progress. Default TRUE.
#' @param v.every Print frequency (every v.every outer iterations). Default 10.
#' @param Lmax Clipping bound for Lagrange multipliers. Default 20.
#' @param c1 Multiplicative factor for rho increase (must be > 1). Default 1.05.
#' @param c2 Threshold for rho update (must be in (0,1)). Default 0.25.
#' @param nstart Number of random starts. Default 1 (no random restarts).
#'   When \code{nstart > 1}, each start uses a random orthogonal rotation of \code{A}
#'   and the solution with the smallest objective value is returned.
#' @param seed Optional integer seed for reproducibility of random starts. Default NULL.
#' @param ncores Number of parallel workers. Default 1 (sequential).
#'   Requires the \code{future} and \code{future.apply} packages for \code{ncores > 1}.
#'   If NULL, defaults to \code{ncores = future::availableCores() - 2}.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{B}: Estimated factor loading matrix (follows a bi-factor structure)
#'   \item \code{Phi}: Estimated factor correlation matrix (follows a bi-factor structure)
#'   \item \code{obj.end}: objective function evaluated at rotated solution B
#'   \item \code{cons.end}: Value of the constraint at solution B
#'   \item \code{rho.end}: Final value of rho
#'   \item \code{outer.iter.end}: Number of outer iterations at convergence
#'   \item \code{conv}: Logical; TRUE if converged before maxit.ou
#'   \item \code{nstart}: Number of random starts used
#'   \item \code{all.obj}: Vector of objective values from all starts (only when \code{nstart > 1})
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(1234)
#' A <- matrix(0, 20, 4) # A 20 x 4 Factor loading matrix
#' A[,1] = runif(20, 1, 2) # Bi-factor structure, first factor
#' A[,2] = runif(20, 0.5, 1) * rbinom(20, 1, 0.25)
#' A[,3] = runif(20, 0.5, 1) * rbinom(20, 1, 0.25)
#' A[,4] = runif(20, 0.5, 1) * rbinom(20, 1, 0.25)
#'
#' # Create rotation matrix (via random matrix):
#' Ah =  array(rnorm(length(A), sd = .5), dim = dim(A))
#' Tr = eigen(t(Ah) %*% Ah)$vectors
#' Ah = (A)%*%(Tr)
#'
#' # Single start
#' res <- bifacL1rot::bifactorL1(Ah)
#'
#' # Multiple random starts in parallel
#' res <- bifacL1rot::bifactorL1(Ah, nstart = 10, ncores = 4, seed = 1234)
#' }
#'
#' @export
#' @useDynLib bifacL1rot, .registration = TRUE
#' @importFrom Rcpp sourceCpp
bifactorL1 <- function(A, Bstart = NULL, Phi = NULL, rho = 1, t = 1/1000,
                maxit.ou = 5000, maxit.in = 300, orthogonal = FALSE,
                tol1 = 1e-6, tol2 = 1e-4, verbose = TRUE, v.every = 10L,
                Lmax = 20, c1 = 1.05, c2 = 0.25,
                nstart = 1L, seed = NULL, ncores = 1) {

    # Input validation
    if (!is.matrix(A)) A <- as.matrix(A)
    if (!is.null(Bstart) && !is.matrix(Bstart)) Bstart <- as.matrix(Bstart)
    if (!is.null(Phi) && !is.matrix(Phi)) Phi <- as.matrix(Phi)

    maxit.ou = as.integer(maxit.ou)
    maxit.in = as.integer(maxit.in)
    v.every = as.integer(v.every)
    nstart = as.integer(nstart)
    ncores = as.integer(ncores)
    if (nstart < 1L) stop("nstart must be >= 1")

    # Auto-detect cores when nstart > 1 and ncores not specified
    if (nstart > 1L && is.null(ncores)) {
        if (requireNamespace("future", quietly = TRUE)) {
            ncores <- future::availableCores() - 2
        } else {
            ncores <- 1L
        }
    } else {
        ncores <- if (is.null(ncores)) 1L else as.integer(ncores)
    }
    if (ncores < 1L) stop("ncores must be >= 1")

    # --- Single start (original behavior) ---
    if (nstart == 1L) {
        result <- ALM_cpp(
            A = A,
            Bstart_ = Bstart,
            Phi_ = Phi,
            rho = rho,
            t = t,
            maxit_ou = maxit.ou,
            maxit_in = maxit.in,
            orthogonal = orthogonal,
            tol1 = tol1,
            tol2 = tol2,
            verbose = verbose,
            v_every = v.every,
            Lmax = Lmax,
            c1 = c1,
            c2 = c2
        )
        result$nstart = 1L
        return(result)
    }

    # --- Multiple random starts ---
    if (!is.null(Bstart)) {
        warning("Bstart is ignored when nstart > 1; random rotations of A are used instead.")
    }

    # Pre-generate seeds for reproducibility
    if (!is.null(seed)) set.seed(seed)
    seeds = sample.int(.Machine$integer.max, nstart)

    # Helper: generate one random start matrix
    make_Bstart = function(A, seed_i) {
        set.seed(seed_i)
        noise = array(rnorm(length(A), sd = 0.1), dim = dim(A))
        Tr = eigen(crossprod(noise), symmetric = TRUE)$vectors
        A %*% Tr
    }

    # Worker function for a single start
    run_one = function(seed_i) {
        Bs = make_Bstart(A, seed_i)
        ALM_cpp(
            A = A,
            Bstart_ = Bs,
            Phi_ = Phi,
            rho = rho,
            t = t,
            maxit_ou = maxit.ou,
            maxit_in = maxit.in,
            orthogonal = orthogonal,
            tol1 = tol1,
            tol2 = tol2,
            verbose = FALSE,
            v_every = v.every,
            Lmax = Lmax,
            c1 = c1,
            c2 = c2
        )
    }

    # Run starts (parallel or sequential)
    if (ncores > 1L) {
        if (!requireNamespace("future", quietly = TRUE) ||
            !requireNamespace("future.apply", quietly = TRUE)) {
            stop("Packages 'future' and 'future.apply' are required for parallel execution.\n",
                 "Install with: install.packages(c('future', 'future.apply'))")
        }
        old_plan = future::plan()
        on.exit(future::plan(old_plan), add = TRUE)
        future::plan(future::multisession, workers = ncores)
        results = future.apply::future_lapply(seeds, run_one, future.seed = FALSE)
    } else {
        results = lapply(seeds, run_one)
    }

    # Select best result (lowest objective)
    obj_vals = vapply(results, function(r) r$obj.end, numeric(1))
    best_idx = which.min(obj_vals)
    best = results[[best_idx]]

    if (verbose) {
        message(sprintf("Random starts: %d | Best obj: %.4f (start %d) | Range: [%.4f, %.4f]",
                        nstart, obj_vals[best_idx], best_idx,
                        min(obj_vals), max(obj_vals)))
    }

    best$nstart = nstart
    best$all.obj = obj_vals
    return(best)
}
