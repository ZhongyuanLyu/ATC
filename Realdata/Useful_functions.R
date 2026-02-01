library(lpSolve)
effective_hamming_error <- function(cluster1, cluster2) {
  mapping_to_list <- function(K, clust) {
    map <- vector("list", K)
    map[[1]] <- c(map[[1]], 1)
    ind <- 1
    for (i in 2:n) {
      for (j in 1:(i - 1)) {
        find <- 0
        if (clust[i] == clust[j]) {
          for (k in 1:ind) {
            if (j %in% map[[k]]) {
              map[[k]] <- c(map[[k]], i)
              find <- 1
              break
            }
          }
        }
        if (find == 1) break
        if (j == (i - 1)) {
          ind <- ind + 1
          map[[ind]] <- c(map[[ind]], i)
        }
      }
    }
    return(map)
  }
  clust1 <- unclass(as.ordered(cluster1))
  clust2 <- unclass(as.ordered(cluster2))
  if ((n <- length(clust1)) != length(clust2)) {
    warning("error: length not equal")
    return
  }
  if ((K <- length(table(clust1))) != length(table(clust2))) {
    warning("the number of clusters are not equal")
    return
  }
  list1 <- mapping_to_list(K, clust1)
  list2 <- mapping_to_list(K, clust2)
  cost_mat <- matrix(0, nrow = K, ncol = K)
  for (i in 1:K) {
    for (j in 1:K) {
      cost_mat[i, j] <- sum(!(list1[[i]] %in% list2[[j]]))
    }
  }
  error <- lp.assign(cost_mat)
  return(error$objval / n)
}

eigen_gap_g <- function(adjMat, covMat, lam_list, r){
  num <- length(lam_list)
  g_list <- rep(0, num)
  for (i in 1:num){
    X = adjMat + lam_list[i] * covMat %*% t(covMat)
    svd_X <- svd(X)
    g_list[i] <- (svd_X$d[r]-svd_X$d[r+1])/svd_X$d[r]
  }
  return(g_list)
}


compute_bandwidth <- function(X) {
  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }
  d <- ncol(X)
  n <- nrow(X)
  dmat <- as.matrix(dist(X))
  qi <- apply(dmat, 1, function(row_distances) {
    non_self <- row_distances[row_distances > 0]
    quantile(non_self, probs = 0.10, na.rm = TRUE)
  })
  q95 <- quantile(qi, probs = 0.95, na.rm = TRUE)
  chi95 <- qchisq(0.95, df = d)
  w <- q95 / sqrt(chi95)
  return(w)
}



