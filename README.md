
# bifacLpRot

<!-- badges: start -->
<!-- badges: end -->

$L_p$ Rotation of Exploratory Generalized Bi-Factor Models

## Installation

You can install the development version of `bifacLpRot` from [GitHub](https://github.com/) with:

```r
# install.packages("devtools")
devtools::install_github("ccardehu/bifacLpRot")
```

## Example

This is a basic example:

``` r
set.seed(1234)
A <- matrix(0, 20, 4) # A 20 x 4 Factor loading matrix
A[,1] = runif(20, 1, 2) # Bi-factor structure, strong first-order factors
A[,2] = runif(20, 0.5, 1) * rbinom(20, 1, 0.25) # Sparse second-order factors
A[,3] = runif(20, 0.5, 1) * rbinom(20, 1, 0.25)
A[,4] = runif(20, 0.5, 1) * rbinom(20, 1, 0.25)
# Create rotation matrix (via random matrix):
Ah =  array(rnorm(length(A), sd = .1), dim = dim(A))
Tr = eigen(t(Ah) %*% Ah)$vectors
Ah = (A)%*%(Tr)
res = bifacLpRot::bifactorLp(Ah)
Arot = res$B
Phi = res$Phi
```

