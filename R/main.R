#' Lp rotation for Generalized Bi-factor Models
#'
#' @param A Input factor loading matrix (J x K)
#' @param Phi Optional estimated correlation matrix (if A comes from oblique rotation). Defaults to identity.
#' @param B_start Optional starting value for B matrix. Defaults to A.
#'   Ignored when \code{nstart > 1}.
#' @param Phi_start Optional starting correlation matrix. Defaults to identity.
#' @param rho Initial penalty parameter for augmented Lagrangian method (L). Default 10.
#' @param t Initial step size. Default 1e-3.
#' @param maxit_ou Maximum outer iterations. Default 5000.
#' @param maxit_in Maximum inner iterations. Default 300.
#' @param maxit_bt Maximum back-tracking iterations. Default 20.
# @param hesit Iteration number to add Hessian information. Default 50.
#' @param orthogonal Logical; if TRUE, constrains factors to be orthogonal. Default FALSE.
#' @param tol1 Convergence tolerance for successive parameter change (outer loop). Default 1e-6.
#' @param tol2 Convergence tolerance for constraint violation check. Default 1e-3.
#' @param tol3 Convergence tolerance for successive parameter change (inner loop). Default 1e-3.
#' @param verbose Logical; print progress. Default TRUE.
#' @param v_every Print frequency (every v_every outer iterations). Default 10.
#' @param Lmax Clipping bound for Lagrange multipliers. Default 20.
#' @param c1 Multiplicative factor for rho increase (must be > 1). Default 1.1.
#' @param c2 Threshold for rho update (must be in (0,1)). Default 0.25.
#' @param p Exponent in Qp (must be in (0,1]). Closed form solutions for 1/2, 2/3, 1. Default 1.
#' @param nstart Number of random starts. Default 1 (no random restarts).
#'   When \code{nstart > 1}, each start uses a random orthogonal rotation of \code{A}
#'   and the solution with the smallest objective value is returned.
#' @param ostart Logical; orthogonal rotation for random starts? Default TRUE.
#' @param seed Optional integer seed for reproducibility of random starts. Default NULL.
#' @param ncores Number of parallel workers. Default 1 (sequential).
#'   Requires the \code{future} and \code{future.apply} packages for \code{ncores > 1}.
#'   If NULL, defaults to \code{ncores = future::availableCores() - 2}.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{B}: Estimated factor loading matrix (follows a bi-factor structure)
#'   \item \code{Phi}: Estimated factor correlation matrix (follows a bi-factor structure)
#'   \item \code{obj.end}: objective function Qp(B) evaluated at rotated solution (B,Phi)
#'   \item \code{cons.end}: Value of the constraint h(B,Phi) at solution (B,Phi)
#'   \item \code{tol.end}: Final value of convergence criterion (parameter difference)
#'   \item \code{rho.end}: Final value of rho
#'   \item \code{iter.end}: Number of outer iterations at convergence
#'   \item \code{converged}: Logical; TRUE if converged before maxit_ou
#'   \item \code{time}: Wall clock computation time (in seconds)
#'   \item \code{nstart}: Number of random starts used
#'   \item \code{all.obje}: Vector of objective values Qp(B) from all starts (only when \code{nstart > 1})
#'   \item \code{all.cons}: Vector of constraint values h(B,Phi) from all starts (only when \code{nstart > 1})
#'   \item \code{all.conv}: Vector of boolean for convergence of all starts (only when \code{nstart > 1})
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(1234)
#' A <- matrix(0, 20, 4) # A 20 x 4 Factor loading matrix
#' A[,1] = runif(20, 0, 1) # Bi-factor structure, first factor
#' A[,2] = runif(20, 0, 1) * rbinom(20, 1, 1/3)
#' A[,3] = runif(20, 0, 1) * rbinom(20, 1, 1/3)
#' A[,4] = runif(20, 0, 1) * rbinom(20, 1, 1/3)
#'
#' # Create orthogonal bi-factor rotation matrix (via random matrix):
#' Tr = qr.Q(qr(matrix(rnorm((ncol(A)-1) * (ncol(A)-1)), ncol(A)-1, ncol(A)-1)))
#' Tr = rbind(c(1, rep(0, ncol(A)-1)), cbind(rep(0, ncol(A)-1), Tr))
#' Ah = (A)%*%(Tr)
#'
#' # Single start
#' res <- bifacLpRot::bifactorLp(Ah)
#'
#' # Multiple random starts in parallel
#' res <- bifacLpRot::bifactorLp(Ah, nstart = 10, ncores = 4, seed = 1234)
#' }
#'
#' @export
#' @useDynLib bifacLpRot, .registration = TRUE
#' @importFrom Rcpp sourceCpp
bifactorLp <- function(A, Phi = NULL, B_start = NULL, Phi_start = NULL,
                       rho = 10, t = 1e-3,
                       maxit_ou = 5000, maxit_in = 300, maxit_bt = 20, #hesit = 50,
                       orthogonal = FALSE,
                       tol1 = 1e-6, tol2 = 1e-3, tol3 = 1e-3,
                       verbose = TRUE, v_every = 10L,
                       Lmax = 20, c1 = 1.1, c2 = 0.25, p = 1,
                       nstart = 1L, ostart = TRUE, seed = NULL, ncores = 1,
                       refine = T) {

    # Input validation
    if(is.null(A)) stop("Factor loading matrix A must be included.")
    if (!is.matrix(A)) A <- as.matrix(A)
    if (!is.null(Phi) && !is.matrix(Phi)) Phi <- as.matrix(Phi)
    if (!is.null(B_start) && !is.matrix(B_start)) B_start <- as.matrix(B_start)
    if (!is.null(Phi_start) && !is.matrix(Phi_start)) Phi_start <- as.matrix(Phi_start)

    maxit_ou = as.integer(maxit_ou)
    maxit_in = as.integer(maxit_in)
    maxit_bt = as.integer(maxit_bt)
    # hesit = as.integer(hesit)
    v_every = as.integer(v_every)
    nstart = as.integer(nstart)
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

    tic = Sys.time()

    idmf = which.max(colSums(A^2))
    idgf = seq(length.out = ncol(A))[seq(length.out = ncol(A)) != idmf]
    A = A[,c(idmf,idgf)]
    if (!is.null(Phi)) Phi = Phi[c(idmf,idgf),c(idmf,idgf)]
    if (!is.null(B_start)) B_start = B_start[,c(idmf,idgf)]
    if (!is.null(Phi_start)) Phi_start = Phi_start[c(idmf,idgf),c(idmf,idgf)]

    # --- Single start ---
    if (nstart == 1L) {
        result <- ALM_cpp(
            A0_ = A,
            Phi0_ = Phi,
            Bstart_ = B_start,
            Phistart_ = Phi_start,
            rho = rho,
            t = t,
            maxit_ou = maxit_ou,
            maxit_in = maxit_in,
            maxit_bt = maxit_bt,
            # hesit = hesit,
            orthogonal = orthogonal,
            tol1 = tol1,
            tol2 = tol2,
            tol3 = tol3,
            verbose = verbose,
            v_every = v_every,
            Lmax = Lmax,
            c1 = c1,
            c2 = c2,
            p = p
        )
        result$nstart = 1L
        return(result)
    }

    # --- Multiple random starts ---
    if (is.null(B_start)) {
        message(sprintf("B_start is NULL; (nstart = %d) random orthogonal bi-factor rotations of initial factor loading matrix are used as starting points.", nstart))
        B_start = as.matrix(A)
    }

    # Pre-generate seeds for reproducibility
    if (!is.null(seed)) set.seed(seed)
    seeds = sample.int(.Machine$integer.max, nstart)

    # Helper: generate one random start matrix
    make_Bstart = function(A, seed_i, ostart) {
        set.seed(seed_i)
        K = ncol(A)
        Z = matrix(rnorm((K-1) * (K-1)), K-1, K-1)
        if(ostart){
            qrZ = qr(Z)
            Tr = qr.Q(qrZ)
            # Sign-correction for Haar distribution
            d = sign(diag(qr.R(qrZ)))
            d[d == 0] = 1
            Tr = Tr * rep(d, each = K-1)
            Tr = rbind(c(1, rep(0, K-1)), cbind(rep(0, K-1), Tr))
        } else {
            d = sqrt(colSums(Z^2)) #sqrt(colSums(Z^2))
            Tr = Z /d #%*% diag(1/d)
            Tr = rbind(c(1, rep(0, K-1)), cbind(rep(0, K-1), Tr))
        }
        A %*% Tr
    }

    # Pre-generate Starting matrices
    Bstart_list = lapply(seeds, function(s) make_Bstart(B_start, s, ostart))

    # Worker function for a single start
    run_one = function(Bs){ # seed_i
        tryCatch({
            # Bs = make_Bstart(B_start, seed_i)
            ALM_cpp(
                A0_ = A,
                Phi0_ = Phi,
                Bstart_ = Bs,
                Phistart_ = Phi_start,
                rho = rho,
                t = t,
                maxit_ou = maxit_ou,
                maxit_in = maxit_in,
                maxit_bt = maxit_bt,
                # hesit = hesit,
                orthogonal = orthogonal,
                tol1 = tol1,
                tol2 = tol2,
                tol3 = tol3,
                verbose = FALSE,
                v_every = v_every,
                Lmax = Lmax,
                c1 = c1,
                c2 = c2,
                p = p
            )
        },
        error = function(e) {
            # warning(sprintf("Seed %d failed: %s", seed_i, conditionMessage(e)))
            warning(sprintf("Start failed: %s", conditionMessage(e)))
            return(NULL)
        })
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
        # results = future.apply::future_lapply(seeds, run_one, future.seed = FALSE) # future.seed = NULL
        results = future.apply::future_lapply(Bstart_list, run_one, future.seed = NULL)
    } else {
        # results = lapply(seeds, run_one)
        results = lapply(Bstart_list, run_one)
    }

    results = Filter(Negate(is.null), results)
    if (length(results) == 0L) {
        stop("All random starts failed.")
    }

    # Select best result (lowest objective; Qp + constraint violation)
    obj_vals = vapply(results, function(r) r$obj.end, numeric(1))
    con_vals = vapply(results, function(r) r$cons.end, numeric(1))
    conv_all = vapply(results, function(r) r$converged, numeric(1))
    best_idx = which.min(obj_vals)
    best = results[[best_idx]]

    if (verbose) {
        message(sprintf("Random starts: %d | Min. Qp(B): %.3f; h(B,Phi): %.3f (start %d) | Range: [%.3f, %.3f]",
                        nstart, obj_vals[best_idx], con_vals[best_idx], best_idx, min(obj_vals), max(obj_vals)))
    }

    if (!best$converged & refine){
        ref = ALM_cpp(
            A0_ = A,
            Phi0_ = Phi,
            Bstart_ = best$B,
            Phistart_ = best$Phi,
            rho = rho,
            t = t,
            maxit_ou = maxit_ou,
            maxit_in = maxit_in,
            maxit_bt = maxit_bt,
            # hesit = hesit,
            orthogonal = orthogonal,
            tol1 = tol1,
            tol2 = tol2,
            tol3 = tol3,
            verbose = FALSE,
            v_every = v_every,
            Lmax = Lmax,
            c1 = c1,
            c2 = c2,
            p = p
        )
        ref$nstart = nstart + 1L
        toc = Sys.time()
        ref$all.obje = c(obj_vals, ref$obj.end)
        ref$all.cons = c(con_vals, ref$cons.end)
        ref$all.conv = c(conv_all, ref$converged)
        ref$time = difftime(toc,tic,units = "secs")
        if(verbose) {
            message(sprintf("Refined from best solution: Qp(B): %.3f; h(B,Phi): %.3f",
                            ref$obj.end, ref$cons.end))
        }
        return(ref)
    }

    toc = Sys.time()
    best$nstart = nstart
    best$all.obje = obj_vals
    best$all.cons = con_vals
    best$all.conv = conv_all
    best$time = difftime(toc,tic,units = "secs")
    return(best)
}
