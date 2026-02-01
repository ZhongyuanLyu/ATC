library(NAC)
library(kernlab)
library(aricode)
source("Useful_functions.R")
set.seed(123)
adjMat <- as.matrix(read.csv("LawyersData/adjMat.csv", header = FALSE))
covMat <- as.matrix(read.csv("LawyersData/covMat.csv", header = FALSE))
labels <- read.csv("LawyersData/label.csv", header = FALSE)[[1]]+1
adjMat <- unname(adjMat)


CASC_res <- CAclustering(adjMat, covMat, K = 2)
NAC_res <- NAC(adjMat, covMat, K = 2)
kernelMat <- kernelMatrix(rbfdot(sigma = 1/(2*compute_bandwidth(covMat))), x = covMat, y = NULL)
lam_list <- seq(0,5, length.out = 20)
eigen_list <- eigen_gap_g(adjMat, kernelMat, lam_list, r = 2)
SDP_res <- SDP(adjMat, kernelMat, lambda = lam_list[which.max(eigen_list)], K = 2, alpha = 1e6, 
               rho = 0.25 , TT = 100, tol = 5)

SDP_mat_res <- matrix(NA, nrow = 100, ncol = 66)
SDP_err <- rep(NA, 100)
for (rep in 1:100) {
  lam_list <- seq(0,5, length.out = 20)
  eigen_list <- eigen_gap_g(adjMat, kernelMat, lam_list, r = 2)
  SDP_mat_res[rep, ] <- SDP(adjMat, kernelMat, lambda = lam_list[which.max(eigen_list)], K = 2, alpha = 1e6, 
                            rho = 0.25 , TT = 100, tol = 5)
  SDP_err[rep] <- effective_hamming_error(labels, SDP_mat_res[rep, ])
}
effective_hamming_error(labels, SDP_res)
effective_hamming_error(labels, CASC_res)
effective_hamming_error(labels, NAC_res)


mean(SDP_err)
ARI(labels, SDP_res)
ARI(labels, CASC_res)
ARI(labels, NAC_res)
