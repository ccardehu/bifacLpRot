#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Non-smooth part of the Augmented Lagrangian
double Qp(arma::mat& B, double p) {
    return arma::accu(arma::pow(arma::abs(B.cols(1, B.n_cols - 1)), p));
}

// Smooth part of the Augmented Lagrangian
double obj2(arma::mat& B, arma::mat& R, arma::mat& L,
            const arma::mat& AAt, double rho1) {

    arma::mat Phi = R * R.t();
    arma::mat resid = AAt - B * Phi * B.t();
    // arma::mat tmp1 = L.t() * resid;

    double term1 = 0.5 * rho1 * arma::accu(arma::square(resid));
    double term2 = arma::accu(L % resid);

    return term1 + term2 ;
}

arma::mat gradB(arma::mat& B, arma::mat& R, arma::mat& L,
                const arma::mat& AAt, double rho1){
    // if (arma::abs(B).max() > 1e10) {
    //     Rcpp::stop("B update diverged. Reduce penalty (rho) or step size (t)");
    // }
    arma::mat Phi = R*R.t();
    arma::mat BP = B*Phi;
    arma::mat dB = -2.0 * rho1 * (AAt - BP*B.t())*BP - 2.0*L*BP ;
    if(!dB.is_finite()){
        Rcpp::stop("Non-finite gradient: dB");
    }
    return(dB);
}

// arma::vec hessB(arma::mat& B, arma::mat& R, arma::mat& L,
//                 const arma::mat& AAt, const arma::mat& Kpq, double rho) {
//
//     arma::mat Phi = R*R.t();
//     arma::mat BP = B*Phi;
//     arma::vec d1 = arma::diagvec(rho * (BP * B.t() - AAt) - L);
//     arma::vec M1 = 2.0 * arma::repmat(d1, R.n_rows, 1);
//     arma::vec d2 = arma::diagvec(BP.t() * BP);
//     arma::vec M2 = 2.0 * rho * arma::repelem(d2, B.n_rows, 1);
//     arma::vec M3 = 2.0 * rho * arma::diagvec(arma::kron(BP.t(), BP) * Kpq);
//     arma::vec dB2 = M1 + M2 + M3;
//     if(!dB2.is_finite()){
//         Rcpp::stop("Non-finite gradient: dB");
//     }
//     return (dB2);
// }

arma::mat gradR(arma::mat& B, arma::mat& R, arma::mat& L,
                const arma::mat& AAt, double rho) {

    arma::mat Phi = R*R.t();
    arma::mat BtLB = B.t()*L*B;
    arma::mat BP = B*Phi;
    arma::mat Bt = B.t();
    arma::mat dR = -2.0 * rho * Bt*(AAt - BP*Bt)*B*R - 2.0 * BtLB*R;
    dR = arma::trimatl(dR);
    dR.row(0).zeros();
    dR.col(0).zeros();
    if (!dR.is_finite()) {
        Rcpp::stop("Non-finite gradient: dR");
    }
    return (dR);
}

// arma::mat prox_LpZero(arma::mat X, arma::vec lambda) {
//     arma::mat lam_mat = arma::reshape(lambda, X.n_rows, X.n_cols);
//     arma::mat tau_mat = arma::square(2.0*lam_mat);
//     arma::mat fix_mat = arma::conv_to<arma::mat>::from(arma::abs(X) > tau_mat);
//     return X % fix_mat;
// }

arma::mat prox_LpOne(arma::mat X, arma::vec lambda) {
    arma::mat lam_mat = arma::reshape(lambda, X.n_rows, X.n_cols);
    return arma::sign(X) % arma::clamp(arma::abs(X) - lam_mat, 0.0, arma::datum::inf);
}

arma::mat prox_LpOneHalf(arma::mat X, arma::vec lambda){
    arma::mat lam_mat = arma::reshape(lambda, X.n_rows, X.n_cols);
    arma::mat tau_mat = 1.5*arma::pow(lam_mat,2.0/3.0);
    arma::mat fix_mat = arma::conv_to<arma::mat>::from(arma::abs(X) > tau_mat);
    arma::mat X_safe = arma::abs(X) + (1.0 - fix_mat);

    arma::mat arg_mat = -(3.0 * std::sqrt(3.0) / 4.0) * lam_mat % arma::pow(X_safe, -1.5);
    arg_mat = arma::clamp(arg_mat, -1.0, 1.0);
    arma::mat out = (2.0 / 3.0) * X % (1.0 + arma::cos((2.0 / 3.0) * arma::acos(arg_mat)));
    return out % fix_mat;

}

arma::mat prox_LpTwoThirds(arma::mat X, arma::vec lambda){
    arma::mat lam_mat = arma::reshape(lambda, X.n_rows, X.n_cols);
    arma::mat tau_mat = 2.0 * arma::pow((2.0 / 3.0) * lam_mat, 0.75);
    arma::mat fix_mat = arma::conv_to<arma::mat>::from(arma::abs(X) > tau_mat);
    arma::mat X_safe = arma::abs(X) + (1.0 - fix_mat);

    arma::mat X2 = arma::square(X_safe);
    arma::mat X4 = arma::square(X2);
    arma::mat lam_mat3 = arma::pow(lam_mat, 3.0);

    arma::mat tmp1 = arma::clamp(X4 / 256.0 - 8.0 / 729.0 * lam_mat3, 0.0, arma::datum::inf);
    tmp1 = arma::sqrt(tmp1);

    // arma::pow(x,e) fails when x < 0 and e fractional, split into sign and value
    arma::mat tmp2p = X2 / 16.0 + tmp1;
    arma::mat tmp2m = X2 / 16.0 - tmp1;
    arma::mat t = arma::sign(tmp2p) % arma::pow(arma::abs(tmp2p), 1.0 / 3.0)
        + arma::sign(tmp2m) % arma::pow(arma::abs(tmp2m), 1.0 / 3.0);

    arma::mat sqr2t = arma::sqrt(2.0 * t);
    arma::mat sqr2t_safe = sqr2t + (1.0 - fix_mat); // guard division

    arma::mat inner = arma::clamp(2.0 * X_safe / sqr2t_safe - 2.0 * t,
                                  0.0, arma::datum::inf);
    arma::mat val = sqr2t + arma::sqrt(inner);
    arma::mat out = arma::sign(X) % arma::pow(val, 3.0) / 8.0;

    return out % fix_mat;
}

arma::mat prox_LpGeneral(arma::mat X, arma::vec lambda, double p, double eps = 1e-6){
    arma::mat lam_mat = arma::reshape(lambda, X.n_rows, X.n_cols);
    arma::mat abs_X = arma::abs(X);
    double exp_inv = 1.0 / (2.0 - p);

    // Threshold: tau_{C,p} = ((2-p) / (2(1-p))) * (2*C*(1-p))^{1/(2-p)}
    arma::mat tau_mat = ((2.0 - p) / (2.0 * (1.0 - p))) *
        arma::pow(2.0 * lam_mat * (1.0 - p), exp_inv);

    // Mask: 1 where |x| > threshold, 0 otherwise (returns 0)
    arma::mat fix_mat = arma::conv_to<arma::mat>::from(abs_X > tau_mat);

    // Initialise bisection interval [a, b]
    // a = (C*p*(1-p))^{1/(2-p)}  if |x| < C + 1
    //     |x| - C                if |x| >= C + 1
    arma::mat a_small = arma::pow(lam_mat * p * (1.0 - p), exp_inv);
    arma::mat a_large = abs_X - lam_mat;
    arma::mat use_large = arma::conv_to<arma::mat>::from(abs_X >= lam_mat + 1.0);
    arma::mat a = (1.0 - use_large) % a_small + use_large % a_large;

    // b = |x|
    arma::mat b = abs_X;

    // J'_tau(t) = t + C*p*t^{p-1} - |tau|
    arma::mat Ja = a + lam_mat * p % arma::pow(a, p - 1.0) - abs_X;

    // Fixed number of bisection iterations
    // From Theorem 3.1: error < (C_max + 1) / 2^{n+1}
    double max_lam = lam_mat.max();
    int n_iter = std::max(1, (int)std::ceil(std::log2((max_lam + 1.0) / eps)));

    // Bisection loop (all elements simultaneously)
    arma::mat c(arma::size(X));
    for (int iter = 0; iter < n_iter; iter++) {
        c = (a + b) / 2.0;
        arma::mat Jc = c + lam_mat * p % arma::pow(c, p - 1.0) - abs_X;

        // Where Ja and Jc have opposite signs: root in [a, c] -> update b
        // Otherwise: root in [c, b] -> update a
        arma::mat move_left = arma::conv_to<arma::mat>::from((Ja % Jc) < 0.0);

        b = move_left % c + (1.0 - move_left) % b;
        a = (1.0 - move_left) % c + move_left % a;
        Ja = (1.0 - move_left) % Jc + move_left % Ja;
        arma::mat check = (b - a) / 2.0 ;
        if(check.max() < eps) break;
    }
    c = (a + b) / 2.0;

    // Restore sign and zero out sub-threshold entries
    return arma::sign(X) % c % fix_mat;
}

void ProxL(arma::mat& R) {
    arma::vec row_norms = arma::sqrt(arma::sum(arma::square(R), 1));
    R.each_col() /= row_norms;
}

// void fixPhi(arma::mat& Phi) {
//     Phi.row(0).zeros();
//     Phi.col(0).zeros();
//     Phi.diag().ones();
// }

void ordCol(arma::mat& B){
    arma::rowvec ss = arma::sum(arma::square(B),0);
    arma::uvec idx = arma::sort_index(ss, "descend");
    B = B.cols(idx) ;
}

void fixB_internal(arma::mat& B,
                   arma::mat& R) {
    ordCol(B);
    arma::vec signs(B.n_cols);
    for (arma::uword i = 0; i < B.n_cols; ++i) {
        arma::uword max_idx = arma::abs(B.col(i)).index_max();
        signs(i) = (B(max_idx, i) >= 0) ? 1.0 : -1.0;
        B.col(i) *= signs(i);
    }
    R.each_col() %= signs;
    R.each_row() %= signs.t();
    // fixPhi(R);
    // R.diag().ones();
}

//' Fix sign indeterminacy in factor loadings (USE WITH CARE)
//'
//' @param B Loading matrix (modified in place)
//' @param R Optional correlation matrix (modified in place). Defaults to NULL.
//' @export
// [[Rcpp::export]]
void fixB(arma::mat& B,
          Rcpp::Nullable<Rcpp::NumericMatrix> R = R_NilValue) {

     arma::vec signs(B.n_cols);
     for (arma::uword i = 0; i < B.n_cols; ++i) {
         arma::uword max_idx = arma::abs(B.col(i)).index_max();
         signs(i) = (B(max_idx, i) >= 0) ? 1.0 : -1.0;
         B.col(i) *= signs(i);
     }
     if(R.isNotNull()) {
         Rcpp::NumericMatrix Rmat(R.get());
         arma::mat Ri(Rmat.begin(), Rmat.nrow(), Rmat.ncol(), false, true);
         Ri.each_col() %= signs;
         Ri.each_row() %= signs.t();
         // fixPhi(Ri);
     }
 }

// [[Rcpp::export]]
arma::vec freeR(arma::mat& R) {
    arma::mat sub = R.submat(1, 1, R.n_rows - 1, R.n_cols - 1);
    return sub.elem(arma::trimatl_ind(arma::size(sub)));
}

void bt4B(arma::mat& B, arma::mat& Bn, arma::mat& R, arma::mat& L,
          const arma::mat& AAt, double rho1,
          arma::mat& grad, double& t, double p,
          const std::function<arma::mat(arma::mat, arma::vec)>& ProxB,
          const int maxit_bt = 20, const double delta = 0.1) {

    double fx = obj2(B, R, L, AAt, rho1);
    arma::mat Ph = R * R.t();
    double Qp_old = Qp(B,p);
    arma::vec lam(B.n_elem - B.n_rows);

    for (int i = 0; i < maxit_bt; ++i) {
        lam.fill(t);
        Bn = B - t * grad;
        Bn.cols(1, B.n_cols - 1) = ProxB(Bn.cols(1, B.n_cols - 1), lam);

        // Quadratic upper bound check smooth h(B,Phi) when p == 1 (i.e., convex)
        arma::mat diff = Bn - B;
        double fnew  = obj2(Bn, R, L, AAt, rho1);
        double linear = arma::accu(grad % diff);
        double quad   = (1.0-delta) * arma::accu(arma::square(diff)) / (2.0 * t);
        bool ista_ok = (fnew <= fx + linear + quad);

        // Added check for non-convex settings (p < 1): require descent on Q_p + h
        bool descent_ok = true;
        if (std::abs(p - 1.0) >= 1e-10) {
            double Qp_new = Qp(Bn,p);
            descent_ok = (fnew + Qp_new <= fx + Qp_old + linear + quad);
        }
        if (ista_ok && descent_ok) return;
        t *= 0.5;
    }
}

void bt4R(arma::mat& B, arma::mat& R, arma::mat& Rn, arma::mat& L,
          const arma::mat& AAt, double rho1,
          arma::mat& grad, double& t,
          const int maxit_bt = 20, const double delta = 0.1) {
    double fx = obj2(B, R, L, AAt, rho1);
    for (int i = 0; i < maxit_bt; ++i) {
        Rn = R - t * grad;
        ProxL(Rn);
        arma::mat diff = Rn - R;
        double fnew  = obj2(B, Rn, L, AAt, rho1);
        double linear = arma::accu(grad % diff);
        double quad   = (1.0-delta) * arma::accu(arma::square(diff)) / (2.0 * t);
        if (fnew <= fx + linear + quad) return;
        t *= 0.5;
    }
}

// arma::mat commutation_matrix(int p, int q) {
//     arma::mat K(p * q, p * q, arma::fill::zeros);
//     for (int i = 0; i < p; ++i) {
//         for (int j = 0; j < q; ++j) {
//             K(j * p + i, i * q + j) = 1.0;
//         }
//     }
//     return K;
// }

// [[Rcpp::export]]
Rcpp::List ALM_cpp(Rcpp::Nullable<arma::mat> A0_,
                   Rcpp::Nullable<arma::mat> Phi0_ = R_NilValue,
                   Rcpp::Nullable<arma::mat> Bstart_ = R_NilValue,
                   Rcpp::Nullable<arma::mat> Phistart_ = R_NilValue,
                   double rho = 10,
                   double t = 1e-3,
                   int maxit_ou = 4e3, int maxit_in = 4e2, int maxit_bt = 4e1,
                   bool orthogonal = false,
                   double tol1 = 1e-6, double tol2 = 1e-6, double tol3 = 1e-6,
                   bool verbose = true, int v_every = 10,
                   double c1 = 4, double c2 = 0.25,
                   double p = 1, const double rho_max = 1e6,
                   const double delta = .1) {

    // Input validation
    if (c1 <= 1.0) Rcpp::stop("Fix c1 argument, must be c1 > 1");
    if (c2 <= 0.0 || c2 >= 1.0) Rcpp::stop("Fix c2 argument, must be 0 < c2 < 1");
    if (p <= 0.0 || p > 1.0) Rcpp::stop("Fix p argument, must be 0 < p <= 1");

    // Initialization of B and Phi
    if(A0_.isNull()) Rcpp::stop("Initial matrix A0 is NULL");
    arma::mat A0 = Rcpp::as<arma::mat>(A0_);
    arma::mat Phi0 = Phi0_.isNotNull() ? Rcpp::as<arma::mat>(Phi0_) : arma::eye(A0.n_cols, A0.n_cols);
    arma::mat B = Bstart_.isNotNull() ? Rcpp::as<arma::mat>(Bstart_) : A0;
    arma::mat Phi = Phistart_.isNotNull() ? Rcpp::as<arma::mat>(Phistart_) : arma::eye(B.n_cols, B.n_cols);

    // Loadings and Correlation normalization
    fixB_internal(B, Phi);
    fixB_internal(A0, Phi0);
    arma::mat R = arma::chol(Phi, "lower");

    // Initialize Lagrange multipliers
    arma::mat L(B.n_rows, B.n_rows, arma::fill::zeros);

    // Number of parameters (for scaling)
    int NR = orthogonal ? 0 : (static_cast<int>(freeR(R).n_elem) - 1);
    int NP = static_cast<int>(B.n_elem) + NR;
    // int NJ = B.n_rows * (B.n_rows + 1)/2;

    // const arma::mat Kpq = commutation_matrix(B.n_rows, B.n_cols);
    const arma::mat AAt = A0 * Phi0 * A0.t();
    double outn = Qp(B,p);
    // double froAAt = arma::norm(AAt,"fro");

    if (verbose) {
        Rcpp::Rcout << "\n Qp(B) (iter: 0): " << std::fixed << std::setprecision(3) << outn;
    }

    int i = 0;
    double critR1 = 0, stopC1 = 0;

    // Protect against floating point in p
    std::function<arma::mat(arma::mat, arma::vec)> ProxB;
    if (std::abs(p - 1.0) < 1e-10) {
        ProxB = prox_LpOne;
    } else if (std::abs(p - 0.5) < 1e-10) {
        ProxB = prox_LpOneHalf;
    } else if (std::abs(p - 2.0/3.0) < 1e-10) {
        ProxB = prox_LpTwoThirds;
    } else {
        ProxB = [p](arma::mat X, arma::vec lambda) {
            return prox_LpGeneral(X, lambda, p, 1e-6);
        };
    }

    // Tolerances:
    tol1 = std::max(tol1, std::sqrt(arma::datum::eps));
    tol2 = std::max(tol2, std::sqrt(arma::datum::eps));
    tol3 = std::max(tol3, std::sqrt(arma::datum::eps));
    bool converged = false;
    // arma::vec ihess = arma::ones<arma::vec>(B.n_elem);
    // arma::mat ihgb(arma::size(B));

    for (i = 0; i < maxit_ou; i++) {
        if (i % 10 == 0) Rcpp::checkUserInterrupt();
        double tB = t ;
        double tR = t ;
        arma::mat Bo = B;
        arma::mat Ro = R;
        arma::mat Phio = Ro * Ro.t();

        for (int j = 0; j < maxit_in; j++) {
            if (j % 10 == 0) Rcpp::checkUserInterrupt();
            arma::mat Bn = B;
            arma::mat Rn = R;
            double critR0 = 0.0;

            // in-place backtracking fix for Bn
            arma::mat gradb = gradB(B, R, L, AAt, rho);
            bt4B(B, Bn, R, L, AAt, rho, gradb, tB, p, ProxB, maxit_bt, delta);

            if (!Bn.is_finite()) {
                Rcpp::stop("Check B at iter: %i, inner: %i", i+1, j+1);
            }

            if (!orthogonal) {
                // in-place backtracking fix for Rn
                arma::mat gradr = gradR(B, R, L, AAt, rho);
                bt4R(B, R, Rn, L, AAt, rho, gradr, tR, maxit_bt, delta);

                if (!Rn.is_finite()) {
                    Rcpp::stop("Check R at iter: %i, inner: %i", i+1, j+1);
                }
                critR0 = arma::accu(arma::square(freeR(R) - freeR(Rn)));
            }

            double stopC0 = (arma::accu(arma::square(B - Bn)) + critR0) / NP;
            B = Bn;
            R = Rn;
            if (std::sqrt(stopC0) < tol3) break;
            tB *= 2.0;
            tR *= 2.0;
        }
        Phi = R * R.t();

        // Update Lagrange multipliers
        L += rho * (AAt - B * Phi * B.t());
        L = 0.5 * (L + L.t());
        // L = arma::clamp(L, -Lmax, Lmax);
        // arma::uvec idL = arma::find(arma::abs(L) == Lmax);
        // L(idL).fill(0.0);
        outn = Qp(B,p);

        if (verbose && (i % v_every == 0)) {
            Rcpp::Rcout << "\r Qp(B) (outer iter: " << i+1 << "): " << std::fixed
                        << std::setprecision(3) << outn << std::flush;
        }

        critR1 = orthogonal ? 0.0 : arma::accu(arma::square(freeR(R) - freeR(Ro)));
        stopC1 = (arma::accu(arma::square(B - Bo)) + critR1) / NP;
        double resid_new = arma::norm(AAt - B * Phi * B.t(), "fro");

        if ((std::sqrt(stopC1) < tol1) && (resid_new < tol2)){
            converged = true;
            break;
        }

        // Adaptive rho update
        double resid_old = arma::norm(AAt - Bo * Phio * Bo.t(), "fro");
        if (resid_new > c2 * resid_old) rho = std::min(rho*c1, rho_max);
    }

    // Final sign fix
    fixB_internal(B, Phi);

    if (verbose) {
        Rcpp::Rcout << "\r Qp(B) (outer iter: " << i+1 << "): " << std::fixed
                    << std::setprecision(3) << outn << std::endl;
    }

    return Rcpp::List::create(
        Rcpp::Named("B") = B,
        Rcpp::Named("Phi") = Phi,
        Rcpp::Named("obj.end") = outn,
        Rcpp::Named("cons.end") = arma::norm(AAt - B*Phi*B.t(), "fro"),
        Rcpp::Named("tol.end") = std::sqrt(stopC1),
        Rcpp::Named("rho.end") = rho,
        Rcpp::Named("iter.end") = i,
        Rcpp::Named("converged") = converged
    );
}
