#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

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

    arma::mat Phi = R*R.t();
    arma::mat BP = B*Phi;
    arma::mat dB = -2.0 * rho * (AAt - BP*B.t())*BP - 2.0*L*BP;
    if(!dB.is_finite()){
        Rcpp::stop("Non-finite dB");
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
    return (M1 + M2 + M3);
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
        Rcpp::stop("Check dR");
    }
    return (dR);
}

arma::mat prox_LpOne(arma::mat X, arma::vec lambda) {
    arma::mat lam_mat = arma::reshape(lambda, X.n_rows, X.n_cols);
    return arma::sign(X) % arma::clamp(arma::abs(X) - lam_mat, 0.0, arma::datum::inf);
}

double LpOneHalf(double x, double lam){
    double tau = 1.5 * std::pow(lam, 2.0/3.0);
    if(std::abs(x) <= tau){
        return 0.0;
    }
    double arg = -(3.0 * std::sqrt(3.0) / 4.0) * lam * std::pow(std::abs(x), -1.5);
    if (arg < -1.0) arg = -1.0;
    if (arg >  1.0) arg =  1.0;
    double out = (2.0 / 3.0) * x * (1.0 + std::cos((2.0 / 3.0) * std::acos(arg)));
    return out;
}

arma::mat prox_LpOneHalf(arma::mat X, arma::vec lambda){
    arma::mat lam_mat = arma::reshape(lambda, X.n_rows, X.n_cols);
    arma::mat out(arma::size(X));
    for(arma::uword j = 0; j < X.n_cols; j++){
        for(arma::uword i = 0; i < X.n_rows; i++){
            double tmp1 = X(i,j);
            double tmp2 = lam_mat(i,j);
            out(i,j) = LpOneHalf(tmp1, tmp2);
        }
    }
    return out;
}

double LpTwoThirds(double x, double lam){
    double tau = 2.0 * std::pow(2.0/3.0 * lam, 0.75);
    if(std::abs(x) <= tau){
        return 0.0;
    }
    double x2 = std::pow(x,2.0);
    double x4 = x2*x2;
    double lam3 = std::pow(lam,3.0);

    double tmp1 = x4/256.0 - 8.0/729.0*lam3;
    if(tmp1 < 0.0) tmp1 = 0.0;  // numerical safety
    tmp1 = std::sqrt(tmp1);
    double t = std::cbrt(x2/16.0 + tmp1) + std::cbrt(x2/16.0 - tmp1);

    double sqr2t = std::sqrt(2.0*t);
    double inner = 2.0*std::abs(x)/sqr2t - 2.0*t;
    if(inner < 0.0) inner = 0.0;  // numerical safety

    double val = sqr2t + std::sqrt(inner);
    double out = std::pow(val,3.0)/8.0;
    if(x < 0.0) out = -out;
    return out;
}

arma::mat prox_LpTwoThirds(arma::mat X, arma::vec lambda){
    arma::mat lam_mat = arma::reshape(lambda, X.n_rows, X.n_cols);
    arma::mat out(arma::size(X));
    for(arma::uword j = 0; j < X.n_cols; j++){
        for(arma::uword i = 0; i < X.n_rows; i++){
            double tmp1 = X(i,j);
            double tmp2 = lam_mat(i,j);
            out(i,j) = LpTwoThirds(tmp1, tmp2);
        }
    }
    return out;
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
            arma::mat& grad, arma::vec& ihess, double t,
            const int maxit_bt = 100, const double beta = 0.5, const double c = 1e-4) {

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
            const int maxit_bt = 100, const double beta = 0.5, const double c = 1e-4) {

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
                   double rho = 1.0, double t = 0.001,
                   int maxit_ou = 5000, int maxit_in = 300, int hesit = 50,
                   bool orthogonal = false,
                   double tol1 = 1e-6, double tol2 = 1e-4,
                   bool verbose = true, int v_every = 10,
                   double Lmax = 20.0, double c1 = 1.05, double c2 = 0.25,
                   double p = 1) {

    // Input validation
    if (c1 <= 1.0) Rcpp::stop("Fix c1 argument, must be c1 > 1");
    if (c2 <= 0.0 || c2 >= 1.0) Rcpp::stop("Fix c2 argument, must be 0 < c2 < 1");
    if (p <= 0.0 || p > 1.0) Rcpp::stop("Fix c2 argument, must be 0 < p <= 1");

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

    arma::mat Kpq = commutation_matrix(B.n_rows, B.n_cols);
    const arma::mat AAt = A * Phi0 * A.t();

    double outn = arma::accu(arma::abs(B));
    if (verbose) {
        Rcpp::Rcout << "\n g(B) (iter: 0): " << std::fixed << std::setprecision(3) << outn;
    }

    int i = 0, j = 0;

    for (i = 1; i <= maxit_ou; ++i) {
        double tB = t;
        double tR = t;
        arma::mat Bo = B;
        arma::mat Ro = R;
        arma::mat Phio = Ro * Ro.t();

        for (j = 1; j <= maxit_in; ++j) {
            arma::mat Bn = B;
            arma::mat Rn = R;
            double crit = 0.0;

            arma::mat gradb = gradB(B, R, L, AAt, rho);
            arma::vec ihess;
            arma::mat ihgb;

            if (i > hesit) {
                ihess = 1.0 / hessB(B, R, L, AAt, Kpq, rho);
                tB = bt4B(Bn, Rn, L, AAt, rho, gradb, ihess, tB, maxit_in, 0.5, tol2);
                ihgb = arma::reshape(ihess % arma::vectorise(gradb),
                                     B.n_rows, B.n_cols) * tB;
            } else {
                ihess = arma::ones<arma::vec>(B.n_elem);
                ihgb = gradb * tB;
            }

            // Update first column (no L1 penalty)
            Bn.col(0) = B.col(0) - ihgb.col(0);

            // Update remaining columns with soft thresholding
            arma::mat B_rest = B.cols(1, B.n_cols - 1) - ihgb.cols(1, ihgb.n_cols - 1);
            arma::vec ihess_rest = ihess.subvec(B.n_rows, ihess.n_elem - 1);
            if(p == 1){
                Bn.cols(1, Bn.n_cols - 1) = prox_LpOne(B_rest, tB * ihess_rest);
            } else if (p == 0.50){
                Bn.cols(1, Bn.n_cols - 1) = prox_LpOneHalf(B_rest, tB * ihess_rest);
            } else if (p == 0.66){
                Bn.cols(1, Bn.n_cols - 1) = prox_LpTwoThirds(B_rest, tB * ihess_rest);
            } else {
                Rcpp::stop("Check value of p");
            }

            if (!Bn.is_finite()) {
                Rcpp::stop("Check B at iter: %i, inner: %i", i, j);
            }

            if (!orthogonal) {
                arma::mat gradr = gradR(B, R, L, AAt, rho);
                if (i > hesit) {
                    tR = bt4R(Bn, Rn, L, AAt, rho, gradr, tR, maxit_in, 0.5, tol2);
                }
                arma::mat gR = gradr * tR / NP;
                Rn = R - gR;
                ProxL(Rn);

                if (!Rn.is_finite()) {
                    Rcpp::stop("Check R at iter: %i, inner: %i", i, j);
                }
                crit = arma::norm(freeR(R) - freeR(Rn), 2);
            }

            double stopC0 = (arma::norm(B - Bn, "fro") + crit) / NP;
            B = Bn;
            R = Rn;
            if (stopC0 < tol1) break;
        }

        Phi = R * R.t();

        // Update Lagrange multipliers
        L = arma::clamp(L + rho * (AAt - B * Phi * B.t()), -Lmax, Lmax);
        L = 0.5 * (L + L.t());

        outn = arma::accu(arma::abs(B));

        if (verbose && (i % v_every == 0)) {
            Rcpp::Rcout << "\r g(B) (outer iter: " << i
                        << ", inner iter: " << j << "): " << std::fixed
                        << std::setprecision(3) << outn << std::flush;
        }

        double crit_outer = orthogonal ? 0.0 : arma::norm(freeR(R) - freeR(Ro), 2);
        double stopC1 = (arma::norm(B - Bo, "fro") + crit_outer) / NP;

        if (stopC1 < tol1) break;

        // Adaptive rho update
        double resid_new = arma::norm(AAt - B * Phi * B.t(), "fro");
        double resid_old = arma::norm(AAt - Bo * Phio * Bo.t(), "fro");
        if (resid_new < c2 * resid_old) {
            rho *= c1;
        }
    }

    // Final sign fixing
    fixB_internal(B, Phi);

    if (verbose) {
        Rcpp::Rcout << "\r g(B) (outer iter: " << i
                    << ", inner iter: " << j << "): " << std::fixed
                    << std::setprecision(3) << outn << std::endl;
    }

    return Rcpp::List::create(
        Rcpp::Named("B") = B,
        Rcpp::Named("Phi") = Phi,
        Rcpp::Named("obj.end") = outn,
        Rcpp::Named("cons.end") = arma::norm(AAt - B * Phi * B.t(), "fro"),
        Rcpp::Named("rho.end") = rho,
        Rcpp::Named("outer.iter.end") = i,
        Rcpp::Named("conv") = (i < maxit_ou)
    );
}
