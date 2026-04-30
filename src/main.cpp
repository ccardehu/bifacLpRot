#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

double Qp(arma::mat& B, double p) {
    return arma::accu(arma::pow(arma::abs(B.cols(1, B.n_cols - 1)), p));
}

double obj2(arma::mat& B, arma::mat& R, arma::mat& L,
            const arma::mat& AAt, double rho) {

    arma::mat Phi = R * R.t();
    arma::mat resid = AAt - B * Phi * B.t();
    arma::mat tmp1 = L.t() * resid;

    double term1 = 0.5 * rho * arma::accu(arma::square(resid));
    double term2 = arma::accu(L % resid);

    return term1 + term2;
}

arma::mat gradB(arma::mat& B, arma::mat& R, arma::mat& L,
                const arma::mat& AAt, double rho){
    if (arma::abs(B).max() > 1e2) {
        Rcpp::stop("B update diverged. Reduce penalty (rho) or step size (t)");
    }
    arma::mat Phi = R*R.t();
    arma::mat BP = B*Phi;
    arma::mat dB = -2.0 * rho * (AAt - BP*B.t())*BP - 2.0*L*BP;
    if(!dB.is_finite()){
        Rcpp::stop("Non-finite gradient: dB");
    }
    return(dB);
}

arma::vec hessB(arma::mat& B, arma::mat& R, arma::mat& L,
                const arma::mat& AAt, const arma::mat& Kpq, double rho) {

    arma::mat Phi = R*R.t();
    arma::mat BP = B*Phi;
    arma::vec d1 = arma::diagvec(rho * (BP * B.t() - AAt) - L);
    arma::vec M1 = 2.0 * arma::repmat(d1, R.n_rows, 1);
    arma::vec d2 = arma::diagvec(BP.t() * BP);
    arma::vec M2 = 2.0 * rho * arma::repelem(d2, B.n_rows, 1);
    arma::vec M3 = 2.0 * rho * arma::diagvec(arma::kron(BP.t(), BP) * Kpq);
    arma::vec dB2 = M1 + M2 + M3;
    if(!dB2.is_finite()){
        Rcpp::stop("Non-finite gradient: dB");
    }
    return (dB2);
}

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

arma::mat prox_LpOne(arma::mat X, arma::vec lambda) {
    arma::mat lam_mat = arma::reshape(lambda, X.n_rows, X.n_cols);
    return arma::sign(X) % arma::clamp(arma::abs(X) - lam_mat, 0.0, arma::datum::inf);
}

// double LpOneHalf(double x, double lam){
//     double tau = 1.5 * std::pow(lam, 2.0/3.0);
//     if(std::abs(x) <= tau){
//         return 0.0;
//     }
//     double arg = -(3.0 * std::sqrt(3.0) / 4.0) * lam * std::pow(std::abs(x), -1.5);
//     if (arg < -1.0) arg = -1.0;
//     if (arg >  1.0) arg =  1.0;
//     double out = (2.0 / 3.0) * x * (1.0 + std::cos((2.0 / 3.0) * std::acos(arg)));
//     return out;
// }

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

// double LpTwoThirds(double x, double lam){
//     double tau = 2.0 * std::pow(2.0/3.0 * lam, 0.75);
//     if(std::abs(x) <= tau){
//         return 0.0;
//     }
//     double x2 = std::pow(x,2.0);
//     double x4 = x2*x2;
//     double lam3 = std::pow(lam,3.0);
//
//     double tmp1 = x4/256.0 - 8.0/729.0*lam3;
//     if(tmp1 < 0.0) tmp1 = 0.0;  // numerical safety
//     tmp1 = std::sqrt(tmp1);
//     double t = std::cbrt(x2/16.0 + tmp1) + std::cbrt(x2/16.0 - tmp1);
//
//     double sqr2t = std::sqrt(2.0*t);
//     double inner = 2.0*std::abs(x)/sqr2t - 2.0*t;
//     if(inner < 0.0) inner = 0.0;  // numerical safety
//
//     double val = sqr2t + std::sqrt(inner);
//     double out = std::pow(val,3.0)/8.0;
//     if(x < 0.0) out = -out;
//     return out;
// }

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

    // arma::pow() fails for negative bases with fractional exponents,
    // so split into sign and magnitude
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
    // Evaluate at left endpoint
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

void fixB_internal(arma::mat& B,
                   arma::mat& R) {
    arma::vec signs(B.n_cols);
    for (arma::uword i = 0; i < B.n_cols; ++i) {
        arma::uword max_idx = arma::abs(B.col(i)).index_max();
        signs(i) = (B(max_idx, i) >= 0) ? 1.0 : -1.0;
        B.col(i) *= signs(i);
    }
    for (arma::uword i = 0; i < R.n_cols; ++i) {
        for (arma::uword j = i; j < R.n_rows; ++j) {
            R(j, i) *= signs(i) * signs(j);
            R(i, j) = R(j, i);
        }
    }
    R.diag().ones();
    if(arma::all(signs > 0.0)) R = arma::abs(R);
}

//' Fix sign indeterminacy in factor loadings
//'
//' @param B Loading matrix (modified in place)
//' @param R_ Optional correlation matrix (modified in place). Defaults to identity.
//' @export
// [[Rcpp::export]]
void fixB(arma::mat& B,
          Rcpp::Nullable<Rcpp::NumericMatrix> R_) {

     arma::vec signs(B.n_cols);
     for (arma::uword i = 0; i < B.n_cols; ++i) {
         arma::uword max_idx = arma::abs(B.col(i)).index_max();
         signs(i) = (B(max_idx, i) >= 0) ? 1.0 : -1.0;
         B.col(i) *= signs(i);
     }
     if(R_.isNotNull()) {
         Rcpp::NumericMatrix Rmat(R_.get());
         arma::mat R(Rmat.begin(), Rmat.nrow(), Rmat.ncol(), false, true);

         for (arma::uword i = 0; i < R.n_cols; ++i) {
             for (arma::uword j = i; j < R.n_rows; ++j) {
                 R(j, i) *= signs(i) * signs(j);
                 R(i, j) = R(j, i);
             }
         }
         R.diag().ones();
         if (arma::all(signs > 0.0)) R = arma::abs(R);
     }
 }

void fixPhi(arma::mat& Phi) {
    Phi.row(0).zeros();
    Phi.col(0).zeros();
    Phi.diag().ones();
}

// [[Rcpp::export]]
arma::vec freeR(arma::mat& R) {
    arma::mat sub = R.submat(1, 1, R.n_rows - 1, R.n_cols - 1);
    return sub.elem(arma::trimatl_ind(arma::size(sub)));
}

double bt4B(arma::mat& B, arma::mat& R, arma::mat& L,
            const arma::mat& AAt, double rho,
            arma::mat& grad, double t,
            const int maxit_bt = 20, const double beta = 0.5, const double c = 1e-4) {

    double fx = obj2(B, R, L, AAt, rho);
    double grad_sqnorm = arma::accu(arma::square(grad));
    for (int i = 0; i < maxit_bt; ++i) {
        arma::mat Bnew = B - t * grad;
        double fnew = obj2(Bnew, R, L, AAt, rho);
        if (fnew <= fx - c * t * grad_sqnorm) {
            return t;
        }
        t *= beta;
    }
    return t;
}

double bt4Bh(arma::mat& B, arma::mat& R, arma::mat& L,
             const arma::mat& AAt, double rho,
             arma::mat& grad, arma::vec& ihess, double t,
             const int maxit_bt = 20, const double beta = 0.5, const double c = 1e-4) {

    double fx = obj2(B, R, L, AAt, rho);
    arma::mat iHgrad = arma::reshape(ihess % arma::vectorise(grad), B.n_rows, B.n_cols);
    double iHgrad_sqnorm = arma::accu(arma::square(iHgrad));
    for (int i = 0; i < maxit_bt; ++i) {
        arma::mat Bnew = B - t * iHgrad;
        double fnew = obj2(Bnew, R, L, AAt, rho);
        if (fnew <= fx - c * t * iHgrad_sqnorm) {
            return t;
        }
        t *= beta;
    }
    return t;
}

double bt4R(arma::mat& B, arma::mat& R, arma::mat& L,
            const arma::mat& AAt, double rho,
            arma::mat& grad, double t,
            const int maxit_bt = 20, const double beta = 0.5, const double c = 1e-4) {

    double fx = obj2(B, R, L, AAt, rho);
    double grad_sqnorm = arma::accu(arma::square(grad));

    for (int i = 0; i < maxit_bt; ++i) {
        arma::mat Rnew = R - t * grad;
        ProxL(Rnew);
        double fnew = obj2(B, Rnew, L, AAt, rho);
        if (fnew <= fx - c * t * grad_sqnorm) {
            return t;
        }
        t *= beta;
    }
    return t;
}

// [[Rcpp::export]]
arma::mat commutation_matrix(int p, int q) {
    arma::mat K(p * q, p * q, arma::fill::zeros);
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < q; ++j) {
            K(j * p + i, i * q + j) = 1.0;
        }
    }
    return K;
}

// [[Rcpp::export]]
Rcpp::List ALM_cpp(arma::mat& A,
                   Rcpp::Nullable<arma::mat> Phi0_ = R_NilValue,
                   Rcpp::Nullable<arma::mat> Bstart_ = R_NilValue,
                   Rcpp::Nullable<arma::mat> Phi_ = R_NilValue,
                   double rho = 1.0, double t = 1e-3,
                   int maxit_ou = 5000, int maxit_in = 300, //int hesit = 50,
                   bool orthogonal = false,
                   double tol1 = 1e-6, double tol2 = 1e-6, double tol3 = 1e-4,
                   bool verbose = true, int v_every = 10,
                   double Lmax = 20.0, double c1 = 1.05, double c2 = 0.25,
                   double p = 1) {

    // wall_clock timer;
    // timer.tic();

    // Input validation
    if (c1 <= 1.0) Rcpp::stop("Fix c1 argument, must be c1 > 1");
    if (c2 <= 0.0 || c2 >= 1.0) Rcpp::stop("Fix c2 argument, must be 0 < c2 < 1");
    if (p <= 0.0 || p > 1.0) Rcpp::stop("Fix p argument, must be 0 < p <= 1");

    // Initialization of B and Phi
    arma::mat Phi0 = Phi0_.isNotNull() ? Rcpp::as<arma::mat>(Phi0_) : arma::eye(A.n_cols, A.n_cols);
    arma::mat B = Bstart_.isNotNull() ? Rcpp::as<arma::mat>(Bstart_) : A;
    arma::mat Phi = Phi_.isNotNull() ? Rcpp::as<arma::mat>(Phi_) : arma::eye(B.n_cols, B.n_cols);
    fixPhi(Phi);
    arma::mat R = arma::chol(Phi, "lower");

    // Initialize Lagrange multipliers
    arma::mat L(B.n_rows, B.n_rows, arma::fill::zeros);

    // Number of parameters (for scaling)
    int NR = orthogonal ? 0 : static_cast<int>(freeR(R).n_elem) - 1;
    int NP = static_cast<int>(B.n_elem) + NR;

    // const arma::mat Kpq = commutation_matrix(B.n_rows, B.n_cols);
    const arma::mat AAt = A * Phi0 * A.t();

    double outn = Qp(B,p);
    if (verbose) {
        Rcpp::Rcout << "\n Qp(B) (iter: 0): " << std::fixed << std::setprecision(3) << outn;
    }

    int i = 0, j = 0;
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
    const double eps1 = tol1*tol1;
    const double eps2 = tol2*tol2;
    bool converged = false;

    for (i = 1; i <= maxit_ou; ++i) {
        if (i % 10 == 0) Rcpp::checkUserInterrupt();
        double tB = t ;
        double tR = t ;
        arma::mat Bo = B;
        arma::mat Ro = R;
        arma::mat Phio = Ro * Ro.t();

        for (j = 1; j <= maxit_in; ++j) {
            if (j % 10 == 0) Rcpp::checkUserInterrupt();
            arma::mat Bn = B;
            arma::mat Rn = R;
            double critR0 = 0.0;

            arma::mat gradb = gradB(B, R, L, AAt, rho);
            arma::vec ihess = arma::ones<arma::vec>(B.n_elem);
            arma::mat ihgb(arma::size(B));

            // if (i > hesit) {
            //     ihess /= hessB(B, R, L, AAt, Kpq, rho);
            //     tB = bt4Bh(Bn, Rn, L, AAt, rho, gradb, ihess, tB, maxit_in, 0.5, tol2);
            //     ihgb = arma::reshape(ihess % arma::vectorise(gradb),
            //                          B.n_rows, B.n_cols) * tB;
            // } else {
            tB = bt4B(Bn, Rn, L, AAt, rho, gradb, tB, maxit_in, 0.5, tol2);
            ihgb = gradb * tB;
            // }

            // Update first column (no L1 penalty)
            Bn.col(0) = B.col(0) - ihgb.col(0);

            // Update remaining columns with proximal step
            arma::mat B_rest = B.cols(1, B.n_cols - 1) - ihgb.cols(1, ihgb.n_cols - 1);
            arma::vec ihess_rest = ihess.subvec(B.n_rows, ihess.n_elem - 1);
            Bn.cols(1, Bn.n_cols - 1) = ProxB(B_rest, tB * ihess_rest);

            if (!Bn.is_finite()) {
                Rcpp::stop("Check B at iter: %i, inner: %i", i, j);
            }

            if (!orthogonal) {
                arma::mat gradr = gradR(B, R, L, AAt, rho);
                // if (i > hesit) {
                tR = bt4R(Bn, Rn, L, AAt, rho, gradr, tR, maxit_in, 0.5, tol2);
                // }
                Rn = R - gradr * tR;
                ProxL(Rn);

                if (!Rn.is_finite()) {
                    Rcpp::stop("Check R at iter: %i, inner: %i", i, j);
                }
                critR0 = arma::accu(arma::square(freeR(R) - freeR(Rn)));
            }

            double stopC0 = (arma::accu(arma::square(B - Bn)) + critR0) / NP;
            B = Bn;
            R = Rn;
            if (stopC0 < eps2) break;
        }

        Phi = R * R.t();

        // Update Lagrange multipliers
        L += rho * (AAt - B * Phi * B.t());
        L = 0.5 * (L + L.t());
        L = arma::clamp(L, -Lmax, Lmax);

        outn = Qp(B,p);

        if (verbose && (i % v_every == 0)) {
            Rcpp::Rcout << "\r Qp(B) (outer iter: " << i << "): " << std::fixed
                        << std::setprecision(3) << outn << std::flush;
        }

        critR1 = orthogonal ? 0.0 : arma::accu(arma::square(freeR(R) - freeR(Ro)));
        stopC1 = (arma::accu(arma::square(B - Bo)) + critR1) / NP;

        double resid_new = arma::norm(AAt - B * Phi * B.t(), "fro");
        double resid_old = arma::norm(AAt - Bo * Phio * Bo.t(), "fro");

        if ((stopC1 < eps1) && (resid_new < tol3 * arma::norm(AAt, "fro"))){
            converged = true;
            break;
        }
        // if ((stopC1 < eps1)) break;

        // Adaptive rho update
        if (resid_new < c2 * resid_old) {
            rho *= c1;
        }
    }

    // Final sign fix
    fixB_internal(B, Phi);

    if (verbose) {
        Rcpp::Rcout << "\r Qp(B) (outer iter: " << i << "): " << std::fixed
                    << std::setprecision(3) << outn << std::endl;
    }

    // double time = timer.toc();

    return Rcpp::List::create(
        Rcpp::Named("B") = B,
        Rcpp::Named("Phi") = Phi,
        Rcpp::Named("obj.end") = outn,
        Rcpp::Named("cons.end") = arma::norm(AAt - B * Phi * B.t(), "fro"),
        Rcpp::Named("rho.end") = rho,
        Rcpp::Named("outer.iter.end") = i,
        Rcpp::Named("conv") = converged,
        Rcpp::Named("eps1") = stopC1//,
        // Rcpp::Named("time") = time
    );
}
