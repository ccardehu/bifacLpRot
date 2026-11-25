# Main functions ----

output_bifacLp = function(res, A, p, orthogonal, call) {
    K <- ncol(res$B)
    dimnames(res$B) <- list(rownames(A), c("GF", paste0("SF", seq_len(K - 1))))
    dimnames(res$Phi) <- list(colnames(res$B), colnames(res$B))
    res$p          <- p
    res$orthogonal <- orthogonal
    res$call       <- call
    class(res)     <- "bifacLpRot"
    res
}


#' Lp rotation for Generalized Bi-factor Models
#'
#' @param A Input factor loading matrix (J x K)
#' @param Phi Optional estimated correlation matrix (if A comes from oblique rotation). Defaults to identity.
#' @param B_start Optional starting value for B matrix. Defaults to A.
#'   Ignored when \code{nstart > 1}.
#' @param Phi_start Optional starting correlation matrix. Defaults to identity.
#' @param rho Initial penalty parameter for augmented Lagrangian method (L). Default 10.
#' @param t Initial step size. Default 1e-3.
#' @param maxit_ou Maximum outer iterations. Default 4000.
#' @param maxit_in Maximum inner iterations. Default 400.
#' @param maxit_bt Maximum back-tracking iterations. Default 40.
#' @param orthogonal Logical; if TRUE, constrains factors to be orthogonal. Default FALSE.
#' @param tol1 Convergence tol. for avg. successive parameter change (outer loop). Default 1e-5.
#' @param tol2 Convergence tol. for constraint violation check. Default 1e-5.
#' @param tol3 Convergence tol. for successive parameter change (inner loop). Default 1e-5.
#' @param verbose Logical; print progress. Default FALSE.
#' @param v_every Print frequency (every v_every outer iterations). Default 10.
#' @param c1 Multiplicative factor for rho increase (must be > 1). Default 4.
#' @param c2 Threshold for rho update (must be in (0,1)). Default 0.25.
#' @param p Exponent in Qp (must be in (0,1]). Closed form solutions for 1/2, 2/3, 1. Default 1.
#' @param rho_max Maximum value for the penalty parameter. Default 1e8.
#' @param delta Gap in back-tracking step (must be in (0,1)). Default 0.1.
#' @param nstart Number of random starts. Default 0 (no random restarts).
#'   When \code{nstart > 0}, each start uses a random orthogonal rotation of group-block in \code{A}
#'   and the solution with the smallest objective value is returned.
#' @param ostart Logical; orthogonal rotation for random starts? Default TRUE.
#' @param seed Optional integer seed for reproducibility of random starts. Default NULL.
#' @param ncores Number of parallel workers. Default 1 (sequential).
#'   Requires the \code{future} and \code{future.apply} packages for \code{ncores > 1}.
#'   If NULL, defaults to \code{ncores = future::availableCores() - 2}.
#' @param refine Logical; refine best solution from obtained starting values? Default FALSE.
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
#'   \item \code{all.obje}: Vector of objective values Qp(B) from all starts (only when \code{nstart > 0})
#'   \item \code{all.cons}: Vector of constraint values h(B,Phi) from all starts (only when \code{nstart > 0})
#'   \item \code{all.conv}: Vector of boolean for convergence of all starts (only when \code{nstart > 0})
#'   \item \code{all.feas}: Vector of boolean for feasibility h(B,Phi) < tol2 (only when \code{nstart > 0})
#'   \item \code{all.seed}: Vector of integer for random seeds (only when \code{nstart > 0})
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
                       maxit_ou = 4e3, maxit_in = 4e2, maxit_bt = 4e1,
                       orthogonal = FALSE,
                       tol1 = 1e-5, tol2 = 1e-5, tol3 = 1e-5,
                       verbose = FALSE, v_every = 10L,
                       c1 = 4, c2 = 0.25, p = 1, rho_max = 1e8, delta = .01,
                       nstart = 0L, ostart = TRUE, seed = NULL, ncores = 1) {

    # Input validation ----
    if(is.null(A)) stop("Factor loading matrix A must be included.")
    if (!is.matrix(A)) A <- as.matrix(A)
    if (!is.null(Phi) && !is.matrix(Phi)) Phi <- as.matrix(Phi)
    if (!is.null(B_start) && !is.matrix(B_start)) B_start <- as.matrix(B_start)
    if (!is.null(Phi_start) && !is.matrix(Phi_start)) Phi_start <- as.matrix(Phi_start)

    maxit_ou = as.integer(maxit_ou)
    maxit_in = as.integer(maxit_in)
    maxit_bt = as.integer(maxit_bt)
    v_every = as.integer(v_every)
    nstart = as.integer(nstart)
    if (nstart < 0L) stop("nstart must be >= 0")
    cl = match.call()

    # Auto-detect cores when nstart > 0 and ncores not specified
    if (nstart > 0L && is.null(ncores)) {
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

    # Single start ----
    if (nstart == 0L) {
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
            orthogonal = orthogonal,
            tol1 = tol1,
            tol2 = tol2,
            tol3 = tol3,
            verbose = verbose,
            v_every = v_every,
            c1 = c1,
            c2 = c2,
            p = p,
            rho_max = rho_max,
            delta = delta
        )
        result$nstart = 0L
        return(output_bifacLp(result, A, p, orthogonal, cl))
    }

    # Multiple random starts ----
    if (is.null(B_start)) {
        if(verbose) message(sprintf("B_start is NULL; (nstart = %d) random orthogonal bi-factor rotations of initial factor loading matrix used as starting points.", nstart))
        B_start = as.matrix(A)
    }

    # Pre-generate seeds for reproducibility
    if (!is.null(seed)) set.seed(seed)
    if(nstart > 1) {
        seeds = sample.int(.Machine$integer.max, nstart)
    } else {
        if (is.null(seed)){
            if (verbose) {
                message("seed = NULL and Random starts (nstart) = 1. Setting to default seed = 1234")
            }
            seeds = 1234
        } else {
            seeds = seed
        }
    }

    # Helper: generate one random start matrix
    make_Bstart = function(A, seed_i, ostart) {
        set.seed(seed_i)
        K = ncol(A)-1
        Z = matrix(rnorm((K) * (K)), K, K)
        if(ostart){
            qrZ = qr(Z)
            Tr = qr.Q(qrZ)
            # Sign-correction for Haar distribution
            d = sign(diag(qr.R(qrZ)))
            d[d == 0] = 1
            Tr = Tr * rep(d, each = K)
            Tr = rbind(c(1, rep(0, K)), cbind(rep(0, K), Tr))
        } else {
            d = sqrt(colSums(Z^2)) #sqrt(colSums(Z^2))
            Tr = sweep(Z, 2, sqrt(colSums(Z^2)), "/") # Z /d #%*% diag(1/d)
            Tr = rbind(c(1, rep(0, K)), cbind(rep(0, K), Tr))
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
                orthogonal = orthogonal,
                tol1 = tol1,
                tol2 = tol2,
                tol3 = tol3,
                verbose = FALSE,
                v_every = v_every,
                c1 = c1,
                c2 = c2,
                p = p,
                rho_max = rho_max,
                delta = delta
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

    # Pick min(Qp) only among results meeting feasibility tolerance
    obj_vals = vapply(results, function(r) r$obj.end, numeric(1))
    con_vals = vapply(results, function(r) r$cons.end, numeric(1))
    conv_all = vapply(results, function(r) r$converged, logical(1))

    feas_vals = con_vals #/ froAAt # NJ
    feasible  = feas_vals < tol2

    if (any(feasible)) {
        cand = which(feasible)
        best_idx = cand[which.min(obj_vals[cand])]
    } else {
        # No start met the tolerance: take the least-infeasible one and refine.
        best_idx = which.min(feas_vals)
        if (verbose) {
            message(sprintf(
                "No feasible solution (min h(B,Phi) = %.3f > tol2); selecting least-infeasible start.",
                min(feas_vals)))
        }
    }
    best = results[[best_idx]]

    if (verbose) {
        message(sprintf(
            "Best start: %i (of %i; %.0f%% converged within maxit_ou iterations, %.0f%% feasible)\nQp(B): %.3f; h(B,Phi): %.4f | Range Qp (feasible only): [%s] | Range h (feasible only): [%s]",
            best_idx, nstart, 100 * mean(conv_all), 100 * mean(feasible),
            obj_vals[best_idx], con_vals[best_idx],
            if (any(feasible))
                sprintf("%.3f, %.3f", min(obj_vals[feasible]), max(obj_vals[feasible]))
            else "none feasible",
            if (any(feasible))
                sprintf("%.4f, %.4f", min(con_vals[feasible]), max(con_vals[feasible]))
            else "none feasible"))
    }

    toc = Sys.time()
    best$nstart = nstart
    best$all.obje = obj_vals
    best$all.cons = con_vals
    best$all.conv = conv_all
    best$all.feas = feasible
    best$time = difftime(toc,tic,units = "secs")
    best$seeds = seeds
    return(output_bifacLp(best, A, p, orthogonal, cl))
}


# Methods ----

.format_p <- function(p) {
    known <- c(`1` = 1, `2/3` = 2/3, `1/2` = 1/2)
    hit <- which(abs(p - known) < 1e-10)
    if (length(hit)) names(known)[hit] else format(signif(p, 3))
}

#' Print a bi-factor Lp rotation
#'
#' @param x A \code{bifacLpRot} object returned by \code{bifactorLp()}.
#' @param digits Number of decimal places. Default 3.
#' @param cutoff Loadings with absolute value below \code{cutoff} are shown as
#'   blanks. Default 0.01 for readability.
#' @param colord (Optional) Custom order for columns
#' @param ... Unused.
#' @export
print.bifacLpRot <- function(x, digits = 3L, cutoff = 0.01, colord = NULL, ...) {
    zero_tol <- sqrt(.Machine$double.eps)
    orth <- isTRUE(x$orthogonal)
    L    <- unclass(x$B)
    J    <- nrow(L)
    K    <- ncol(L)
    if(is.null(colord)) colord = 1:K
    Phi  <- if (orth || is.null(x$Phi)) diag(K) else x$Phi
    dimnames(Phi) <- list(colnames(L), colnames(L))

    # --- Convergence header ---
    multi <- isTRUE(x$nstart > 0)
    cat(if (orth) "Orthogonal" else "Oblique",
        " bi-factor Lp rotation (p = ", .format_p(x$p), ")",
        if (!isTRUE(x$converged)) " NOT", " converged",
        if (multi) " at lowest minimum.\n" else ".\n", sep = "")

    # --- Random-start diagnostics ---
    if (multi) {
        n    <- length(x$all.feas)
        feas <- x$all.feas
        conv <- x$all.conv[seq_len(n)]
        obj  <- x$all.obje[seq_len(n)]
        cat("Of ", n, " random starts ", round(100 * mean(conv)), "% converged, ",
            round(100 * mean(feas)), "% feasible", sep = "")
        if (any(feas)) {
            q   <- obj[feas]
            tol <- 1e-3 * max(1, abs(min(q)))     # objective values agree to ~3 digits
            cat(", ", round(100 * sum(abs(q - min(q)) < tol) / n),
                "% at the same lowest minimum.\n", sep = "")
            n_min <- length(unique(round(q / tol)))
            if (n_min > 1)
                cat("Feasible starts converged to ", n_min,
                    " different local minima.\n", sep = "")
        } else {
            cat(".\n")
        }
    }

    # --- Loadings ---
    cat(if (multi) "\nLoadings at lowest minimum:\n" else "\nLoadings:\n")
    fx <- format(round(L[,colord], digits))
    fx[abs(L) < max(cutoff, zero_tol)] <- strrep(" ", nchar(fx[1L]))
    print(fx, quote = FALSE, right = TRUE)

    # --- Variance explained (same definitions as GPArotation) ---
    vx <- diag(Phi[colord,colord] %*% crossprod(L[,colord]))          # equals colSums(L^2) when Phi = I
    names(vx) <- colnames(L[,colord])
    varex <- rbind(`SS loadings` = vx)
    if (orth) {
        varex <- rbind(varex, `Proportion Var` = vx / J)
        if (K > 1) varex <- rbind(varex, `Cumulative Var` = cumsum(vx / J))
    }
    cat("\n")
    print(round(varex, digits))

    n_zero <- sum(abs(L[,colord][, -1, drop = FALSE]) < cutoff)
    cat("\nLoadings within cut-off in group-factor block: ", n_zero, " of ", J * (K - 1),
        "\n", sep = "")

    # --- Phi (oblique only) ---
    if (!orth) {
        cat("\nPhi:\n")
        print(round(Phi[colord,colord], digits))
    }

    # --- Solver diagnostics ---
    cat("\nQp(B): ", format(round(x$obj.end, digits), nsmall = digits),
        " | ||h(B, Phi)||_F: ", format(x$cons.end, digits = 3, scientific = TRUE),
        " | outer iterations: ", x$iter.end,
        " | final rho: ", format(x$rho.end, digits = 3), "\n", sep = "")

    invisible(x)
}

# Auxiliary functions ----

#' Generate a population bi-factor loading matrix
#'
#' @param P Number of items.
#' @param K Total number of factors (1 general + K - 1 group factors).
#' @param type One of "sparse", "cluster", or "mix".
#' @param prop Logical; if TRUE, proportionality constraints are (approximately) satisfied.
#' @param n_simple Minimum number of simple items per group factor, i.e.
#'   items that load on the general factor and on exactly one group factor.
#' @param overlap For type = "mix": number of items shared by each pair of
#'   adjacent blocks.
#' @param p_sparse For type = "sparse": probability that a non-anchor group
#'   loading is non-zero.
#' @param seed Optional integer seed.
#' @return A P x K loading matrix; column 1 is the general factor.
#' @export
sim_Astar <- function(P, K, type = c("sparse", "cluster", "mix"),
                      load_floor = 0.0, load_ceil = 2.0,
                      prop = FALSE, n_simple = 3L, overlap = 2L,
                      p_sparse = 1/3, seed = NULL) {
    type <- match.arg(type)
    G <- K - 1L  # number of group factors
    stopifnot(K >= 2, n_simple >= 1, P >= n_simple * G)
    if (!is.null(seed)) set.seed(seed)

    gen_rng <- if (prop) c(0.5*load_ceil, load_ceil) else c(load_floor, load_ceil)
    grp_rng <- if (prop) c(0.25*load_ceil, 0.5*load_ceil) else c(load_floor, load_ceil)

    # Support of the group-factor block (P x G, logical) ----
    S <- matrix(FALSE, P, G)
    if (type == "sparse") {
        # First n_simple * G rows are anchors: n_simple simple items per group
        anchors <- seq_len(n_simple * G)
        S[cbind(anchors, rep(seq_len(G), each = n_simple))] <- TRUE
        rest <- setdiff(seq_len(P), anchors)
        S[rest, ] <- stats::rbinom(length(rest) * G, 1, p_sparse) == 1
    } else {
        # Contiguous blocks; for "mix", adjacent blocks share `overlap` items
        cuts <- round(seq(0, P, length.out = G + 1))
        ext_lo <- if (type == "mix") ceiling(overlap / 2) else 0L
        ext_hi <- if (type == "mix") floor(overlap / 2)   else 0L
        for (g in seq_len(G)) {
            lo <- max(1L, cuts[g] + 1L - if (g > 1) ext_lo else 0L)
            hi <- min(P,  cuts[g + 1]  + if (g < G) ext_hi else 0L)
            S[lo:hi, g] <- TRUE
        }
    }

    # Check the simple-item requirement ----
    simple   <- rowSums(S) == 1L
    n_simple_g <- colSums(S & simple)
    if (any(n_simple_g < n_simple)) {
        stop(sprintf("Group factor(s) %s have fewer than %d simple items; increase P or reduce overlap.",
                     paste(which(n_simple_g < n_simple), collapse = ", "), n_simple))
    }

    # Draw loadings on the support ----
    A <- matrix(0, P, K)
    A[, 1] <- stats::runif(P, gen_rng[1], gen_rng[2])
    A[, -1][S] <- stats::runif(sum(S), grp_rng[1], grp_rng[2])
    A
}
