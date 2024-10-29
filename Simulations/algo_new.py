import numpy as np
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from scipy.stats import multivariate_normal
from scipy.stats.contingency import crosstab
from scipy.optimize import linear_sum_assignment
from stepmix.stepmix import StepMix
from sparsebm import SBM


def reshape_if_needed(X):
    return X if X.ndim > 1 else X.reshape(-1, 1)
    
# Alignment

def align_labels(Z, Z_benchmark):
    # Z, Z_benchmark: (n, ) vector, entries in {0, 1, ..., K - 1}
    # align Z with Z_benchmark
    cost = - crosstab(Z, Z_benchmark)[1]
    _, map_class = linear_sum_assignment(cost) # class k in Z is matched with class map_class[k] in Z_benchmark
    # print(Z, map_class)
    Z_aligned = np.array([map_class[_] for _ in Z])
    err = len(Z) + np.trace(cost[:, map_class])
    return [Z_aligned, map_class, err]


def Hamming_aligned(Z_1, Z_2):
    [_, _, ans] = align_labels(Z_1, Z_2)
    return ans


def align_paras(type_model, paras, Z_benchmark):
    [Z_aligned, map_class, _] = align_labels(paras['labels'], Z_benchmark) # class k in Z_hat is matched with class map_class[k] in Z_benchmark
    inv_map = np.zeros(len(map_class)).astype(int)
    for (j, k) in enumerate(map_class):
        inv_map[k] = j

    ans = {'labels': Z_aligned, 'weights': paras['weights'][inv_map]}

    if type_model == 'SBM':
        ans['probabilities'] = paras['probabilities'][inv_map][:, inv_map]

    elif type_model == 'GMM':
        ans['means'] = paras['means'][inv_map]
        ans['covariances'] = paras['covariances'][inv_map]

    elif type_model == 'LCM':
        ans['Theta'] = paras['Theta'][:, inv_map]

    return ans

#######################################################

# X-data: GMM
def fit_GMM(X, K, spectral = False):
    # X: (n, d) array of data
    # K: Number of clusters
    X = reshape_if_needed(X)
    if spectral == False:
        gmm = GaussianMixture(n_components = K, covariance_type = 'full', n_init = 10, random_state = 0)
        labels = gmm.fit_predict(X)
        ans = {'labels': labels, 'means': gmm.means_, 'covariances': gmm.covariances_, 'weights': gmm.weights_}
    else:
        svd = TruncatedSVD(n_components = K, random_state = 0)
        svd.fit(X.T)
        ebd = svd.components_.T[:, 0:K]
        kmeans = KMeans(n_clusters = K, n_init = 10, random_state = 0)
        kmeans.fit(ebd)
        labels = kmeans.labels_
        # Get the unique labels
        uni_labels = np.unique(labels)
        means = np.zeros((len(uni_labels), X.shape[1]))
        covariances = np.zeros((len(uni_labels), X.shape[1], X.shape[1]))
        for idx, label in enumerate(uni_labels):
        # Select the data points corresponding to the current label
            X_label = X[labels == label]
            # Calculate the mean of the current component
            means[idx, :] = np.mean(X_label, axis=0)
            # Calculate the covariance of the current component
            covariances[idx, :, :] = np.cov(X_label, rowvar=False)
        weights = np.array((1-np.mean(labels), np.mean(labels)))
        ans = {'labels': labels, 'means': means, 'covariances': covariances, 'weights': weights}
    return ans

# fit 2GMM
def fit_2GMM(X0, X1, K, align = True):
    ans = dict()
    ans['X0'] = fit_GMM(X0, K = K)
    tmp = fit_GMM(X1, K = K)
    if align:  # align Z_hat_0 with Z_hat_1
        ans['X1'] = align_paras('GMM', tmp, ans['X0']['labels'])
    else:
        ans['X1'] = tmp
    return ans

def NLL_GMM(X, paras):
    # output: (n, K)
    n = len(X)
    pi = paras['weights']
    K = len(pi)
    ans = np.zeros((n, K)) - np.log(pi)
    for k in range(K):
        mu, Sigma = paras['means'][k], paras['covariances'][k]
        ans[:, k] -= multivariate_normal.logpdf(X, mean = mu, cov = Sigma, allow_singular = False)
    return ans

def NLL_2GMM(X0, X1, paras):
    tmp1, tmp2 = NLL_GMM(X0, paras['X0']), NLL_GMM(X1, paras['X1'])
    NLL = {'target': tmp1, 'source': tmp2}
    return NLL
    

# A-data: SBM
# A: (n, n) symmetric binary-valued adjacency matrix with zero diagonal
# B: (K, K) symmetric probability matrix
# pi: (K, ) vector of prior probabilities
# Z: (n, ) integer-valued label vector, entries in {0, 1, ..., K - 1}

# negative log-likelihood w.r.t. labels
def NLL_SBM(A, paras):
    # output: (n, K) matrix, A[i, j] = negative log-likelihood of node i being in class j
    Log_pi = np.log( paras['weights'] )
    Log_B = np.log( paras['probabilities'] )
    return - Log_pi - A @ Log_B[ paras['labels'] ]


# estimate (B, pi) given (A, Z)
def SBM_estimate_given_labels(A, Z):
    K = np.max(Z) + 1
    n_hat = np.array([np.sum(Z == _) for _ in range(K)])
    pi_hat = n_hat / len(Z)
    B_hat = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            n_ij = n_hat[i] * n_hat[j] - n_hat[i] * (i == j)
            B_hat[i, j] = np.sum(A[Z == i][:, Z == j]) / n_ij

    return B_hat, pi_hat


# refine Z given (A, B, pi, Z), using marginal MLE
def SBM_refine_labels(A, paras):
    return np.argmin(NLL_SBM(A, paras), axis = 1)


# fit SBM by spectral clustering and iterative refinement
def fit_SBM(A, K, sparse_method = False, num_refinement = 10, matrix = 'adjacency', regularizer = 'null'):
    assert np.sum(np.diag(A)) == 0 # no self-loop allowed
    assert matrix == 'adjacency' or matrix == 'Laplacian'
    assert regularizer == 'null' or matrix == 'mean'

    if sparse_method == True:
        model = SBM(K)
        model.fit(A, symmetric=True)
        Z_hat = model.labels
    else:
        # spectral clustering for initial estimation of labels
        # similarity matrix
        if matrix == 'adjacency':
            L = A.copy()
            if regularizer == 'mean':
                L += np.mean(A)
        elif matrix == 'Laplacian': # normalized Laplacian
            D = np.sum(A, axis = 0)
            if regularizer == 'mean':
                D += np.mean(D)
            D_sqrt_inv = 1 / np.sqrt(D)
            L = (A * D_sqrt_inv).T * D_sqrt_inv
        # spectral clustering
        svd = TruncatedSVD(n_components = K, random_state = 0)
        svd.fit(L)
        ebd = svd.components_.T[:, 0:K]
        kmeans = KMeans(n_clusters = K, n_init = 10, random_state = 0)
        kmeans.fit(ebd)
        Z_hat = kmeans.labels_

    # parameter estimation
    B_hat, pi_hat = SBM_estimate_given_labels(A, Z_hat)
    B_hat_spec = B_hat.copy()
    pi_hat_spec = pi_hat.copy()
    
    # iterative estimation
    for _ in range(num_refinement):
        # refine labels
        Z_tmp = SBM_refine_labels(A, {'weights': pi_hat, 'probabilities': B_hat, 'labels': Z_hat})

        # check if some clusters collapsed
        if np.max(Z_tmp) + 1 < K:
            print('warning: iterative refinement interrupted due to cluster collapse')
            B_hat = B_hat_spec
            pi_hat = pi_hat_spec
            break

        # check if label refinement makes any difference
        if np.sum(Z_hat != Z_tmp) == 0: break # no change, stop refinement

        # refine parameters
        Z_hat = Z_tmp.copy()
        B_hat, pi_hat = SBM_estimate_given_labels(A, Z_hat)

    return {'labels': Z_hat, 'probabilities': B_hat, 'weights': pi_hat}


# fit contextual SBM
def fit_CSBM(X0, X1, K, dist, align = True, **kwargs):
    ans = dict()
    if  dist == ['Network','Gaussian']:
        ans['X1'] = fit_GMM(X1, K = K) # fit GMM
        tmp = fit_SBM(A = X0, K = K, **kwargs) # fit SBM
        if align:  # align Z_hat_A with Z_hat_X
            ans['X0'] = align_paras('SBM', tmp, ans['X1']['labels'])
        else:
            ans['X0'] = tmp
    elif dist == ['Gaussian','Network']:
        tmp = fit_SBM(A = X1, K = K, **kwargs) # fit SBM
        ans['X0'] = fit_GMM(X0, K = K) # fit GMM
        if align:  # align Z_hat_A with Z_hat_X
            ans['X1'] = align_paras('SBM', tmp, ans['X0']['labels'])
        else:
            ans['X1'] = tmp
    return ans

# def fit_CSBM(A, X, K, align = True, **kwargs):
#     ans = dict()
#     ans['X'] = fit_GMM(X = X, K = K) # fit GMM
#     tmp = fit_SBM(A = A, K = K, **kwargs) # fit SBM
#     if align:  # align Z_hat_A with Z_hat_X
#         ans['A'] = align_paras('SBM', tmp, ans['X']['labels'])
#     else:
#         ans['A'] = tmp
#     return ans


def NLL_CSBM(X0, X1, paras, dist):
    if  dist == ['Network','Gaussian']:
        NLL = {'target': NLL_SBM(X0, paras['X0']), 'source': NLL_GMM(X1, paras['X1'])}
    elif dist == ['Gaussian','Network']:
        NLL = {'target': NLL_GMM(X0, paras['X0']), 'source': NLL_SBM(X1, paras['X1'])}
    return NLL

def LCM_estimate_given_labels(X, Z):
    p = X.shape[1]
    K = np.max(Z) + 1
    n_hat = np.array([np.sum(Z == _) for _ in range(K)])
    pi_hat = n_hat / len(Z)
    Theta_hat = np.zeros((p, K))
    for j in range(p):
        for k in range(K):
            Theta_hat[j, k] = np.sum(X[Z == k][:, j]) / n_hat[k]

    return Theta_hat, pi_hat


def NLL_LCM(X, paras):
    # output: (n, K) matrix, A[i, j] = negative log-likelihood of subject i being in class j
    Log_pi = np.log( paras['weights'] )
    Log_Theta = np.log( paras['Theta'] )
    Log_1_minus_Theta = np.log(1-paras['Theta'] )
    # print(X.shape, Log_Theta.shape)
    return - Log_pi - X @ Log_Theta - (1-X) @ Log_1_minus_Theta

def LCM_refine_labels(X, paras):
    return np.argmin(NLL_LCM(X, paras), axis = 1)


def fit_LCM_softEM(X, K, num_refinement = 10):
    model_LCM = StepMix(n_components=K, measurement="binary", n_init = 1, random_state=0, progress_bar=0)
    model_LCM.fit(X)
    posterior = model_LCM.get_mm_df()
    posterior_reset = posterior.reset_index()
    variable_col = posterior_reset['variable']
    # Extract the numeric part from the 'variable' column
    posterior_reset['variable_num'] = variable_col.str.extract(r'(\d+)').astype(int)
    # Sorting the DataFrame based on the extracted numeric part
    posterior_sorted = posterior_reset.sort_values('variable_num').drop(columns='variable_num')
    # Converting the sorted DataFrame to a NumPy array
    # print(posterior_sorted.to_numpy())
    Theta_hat = posterior_sorted.to_numpy()[:,3:].astype(float)
    Z_hat = model_LCM.predict(X)
    pi_hat = model_LCM.get_cw_df().to_numpy()[0]
    return {'labels': Z_hat, 'Theta': Theta_hat, 'weights': pi_hat}


def fit_LCM(X, K, num_refinement = 10):
    # spectral clustering for initial estimation of labels
    svd = TruncatedSVD(n_components = K, random_state = 0)
    svd.fit(X.T)
    ebd = svd.components_.T[:, 0:K]
    kmeans = KMeans(n_clusters = K, n_init = 10, random_state = 0)
    kmeans.fit(ebd)
    Z_hat = kmeans.labels_
    
    # parameter estimation
    Theta_hat, pi_hat = LCM_estimate_given_labels(X,Z_hat)
    Theta_hat_spec = Theta_hat.copy()
    pi_hat_spec = pi_hat.copy()
    
    if np.any((Theta_hat_spec == 1) | (Theta_hat_spec == 0)) == 1:
        print('Warning: iterative refinement interrupted due to 0 or 1 occurred in Theta during initialization')
    else:
        for _ in range(num_refinement):
            # refine labels
            Z_tmp = LCM_refine_labels(X, {'weights': pi_hat, 'Theta': Theta_hat, 'labels': Z_hat})
            # check if some clusters collapsed
            if np.max(Z_tmp) + 1 < K:
                print('warning: iterative refinement interrupted due to cluster collapse')
                Theta_hat = Theta_hat_spec
                pi_hat = pi_hat_spec
                break
        
            # check if label refinement makes any difference
            if np.sum(Z_hat != Z_tmp) == 0: break # no change, stop refinement
        
            # refine parameters
            Z_hat = Z_tmp.copy()
            Theta_tmp, pi_hat = LCM_estimate_given_labels(X, Z_hat)
            
            if np.any((Theta_tmp == 1) | (Theta_tmp == 0)) == 1:
                print('Warning: iterative refinement interrupted due to 0 or 1 occurred in Theta during refinement')
                Theta_hat = Theta_hat_spec
                pi_hat = pi_hat_spec
                break
            else:
                Theta_hat = Theta_tmp.copy()
    
    return {'labels': Z_hat, 'Theta': Theta_hat, 'weights': pi_hat}


def fit_BGM(X0, X1, K, dist, align = True, **kwargs):
    ans = dict()
    if  dist == ['Bernoulli','Gaussian']:
        ans['X1'] = fit_GMM(X1, K = K, **kwargs) # fit GMM
        tmp = fit_LCM_softEM(X = X0, K = K) 
        if align:  # align Z_hat_A with Z_hat_X
            ans['X0'] = align_paras('LCM', tmp, ans['X1']['labels'])
        else:
            ans['X0'] = tmp
    elif dist == ['Gaussian','Bernoulli']:
        tmp = fit_LCM_softEM(X = X1, K = K) 
        ans['X0'] = fit_GMM(X0, K = K, **kwargs) # fit GMM
        if align:  # align Z_hat_A with Z_hat_X
            ans['X1'] = align_paras('LCM', tmp, ans['X0']['labels'])
        else:
            ans['X1'] = tmp
    return ans

def NLL_BGM(X0, X1, paras, dist):
    if  dist == ['Bernoulli','Gaussian']:
        NLL = {'target': NLL_LCM(X0, paras['X0']), 'source': NLL_GMM(X1, paras['X1'])}
    elif dist == ['Gaussian','Bernoulli']:
        NLL = {'target': NLL_GMM(X0, paras['X0']), 'source': NLL_LCM(X1, paras['X1'])}
    return NLL


def fit_MM(X0, X1, K, dist, align = True, **kwargs):
    ans = dict()
    if  dist == ['Network','Gaussian']:
        ans['X1'] = fit_GMM(X1, K = K) # fit GMM
        tmp = fit_SBM(A = X0, K = K, **kwargs) # fit SBM
        if align:  # align Z_hat_A with Z_hat_X
            ans['X0'] = align_paras('SBM', tmp, ans['X1']['labels'])
        else:
            ans['X0'] = tmp
    elif dist == ['Gaussian','Network']:
        tmp = fit_SBM(A = X1, K = K, **kwargs) # fit SBM
        ans['X0'] = fit_GMM(X0, K = K) # fit GMM
        if align:  # align Z_hat_A with Z_hat_X
            ans['X1'] = align_paras('SBM', tmp, ans['X0']['labels'])
        else:
            ans['X1'] = tmp
    elif  dist == ['Bernoulli','Network']:
        ans['X1'] = fit_SBM(X1, K = K, **kwargs)
        tmp = fit_LCM_softEM(X = X0, K = K) 
        if align:  # align Z_hat_A with Z_hat_X
            ans['X0'] = align_paras('LCM', tmp, ans['X1']['labels'])
        else:
            ans['X0'] = tmp
    elif dist == ['Network','Bernoulli']:
        tmp = fit_LCM_softEM(X = X1, K = K) 
        ans['X0'] = fit_SBM(X0, K = K, **kwargs)
        if align:  # align Z_hat_A with Z_hat_X
            ans['X1'] = align_paras('LCM', tmp, ans['X0']['labels'])
        else:
            ans['X1'] = tmp
    elif dist == ['Bernoulli','Gaussian']:
        ans['X1'] = fit_GMM(X1, K = K, **kwargs)
        tmp = fit_LCM_softEM(X = X0, K = K) 
        if align:  # align Z_hat_A with Z_hat_X
            ans['X0'] = align_paras('LCM', tmp, ans['X1']['labels'])
        else:
            ans['X0'] = tmp
    elif dist == ['Gaussian','Bernoulli']:
        tmp = fit_LCM_softEM(X = X1, K = K) 
        ans['X0'] = fit_GMM(X0, K = K, **kwargs) 
        if align:  # align Z_hat_A with Z_hat_X
            ans['X1'] = align_paras('LCM', tmp, ans['X0']['labels'])
        else:
            ans['X1'] = tmp
    elif dist == ['Bernoulli','Bernoulli']:
        tmp = fit_LCM_softEM(X = X1, K = K) 
        ans['X0'] = fit_LCM_softEM(X0, K = K)
        if align:  # align Z_hat_A with Z_hat_X
            ans['X1'] = align_paras('LCM', tmp, ans['X0']['labels'])
        else:
            ans['X1'] = tmp
    return ans

def NLL_MM(X0, X1, paras, dist):
    if  dist == ['Bernoulli','Network']:
        NLL = {'target': NLL_LCM(X0, paras['X0']), 'source': NLL_SBM(X1, paras['X1'])}
    elif dist == ['Network','Bernoulli']:
        NLL = {'target': NLL_SBM(X0, paras['X0']), 'source': NLL_LCM(X1, paras['X1'])}
    elif dist == ['Bernoulli','Gaussian']:
        NLL = {'target': NLL_LCM(X0, paras['X0']), 'source': NLL_GMM(X1, paras['X1'])}
    elif dist == ['Gaussian','Bernoulli']:
        NLL = {'target': NLL_GMM(X0, paras['X0']), 'source': NLL_LCM(X1, paras['X1'])}
    elif dist == ['Bernoulli','Bernoulli']:
        NLL = {'target': NLL_LCM(X0, paras['X0']), 'source': NLL_LCM(X1, paras['X1'])}
    elif dist == ['Network','Gaussian']:
        NLL = {'target': NLL_SBM(X0, paras['X0']), 'source': NLL_GMM(X1, paras['X1'])}
    elif dist == ['Gaussian','Network']:
        NLL = {'target': NLL_GMM(X0, paras['X0']), 'source': NLL_SBM(X1, paras['X1'])}
    return NLL



############################################################

#### Transfer learning

# transfer learning by penalized NLL
def TL_given_NLL(NLL, lbd):
    n, K = NLL['target'].shape
    Z_hat_lbd = []

    for i in range(n):
        tmp = np.zeros((K, K))
        for u in range(K):
            for v in range(K):
                tmp[u, v] = NLL['target'][i, u] + NLL['source'][i, v] + lbd * (u != v)
        
        tmp_min = np.min(tmp, axis = 1)
        Z_hat_lbd.append( np.argmin(tmp_min) )
            
    return np.array(Z_hat_lbd)


def DP_given_NLL(NLL): # data pooling
    Z_hat = np.argmin(NLL['target'] + NLL['source'], axis = 1)      
    return Z_hat



def TL_candidates(X0, X1, K, dist, list_lbd, show = False, **kwargs):
    list_Z_hat = []
    # compute negative log-likelihood functions
    if dist == ['Gaussian','Gaussian']:
        pars_hat = fit_2GMM(X0, X1, K = K)
        NLL = NLL_2GMM(X0, X1, pars_hat)
    elif set(dist) in [{'Network', 'Gaussian'}]:
        # pars_hat = fit_CSBM(X0, X1, K = K, dist = dist, **kwargs)
        # NLL = NLL_CSBM(X0, X1, pars_hat, dist = dist)
        pars_hat = fit_MM(X0, X1, K = K, dist = dist, **kwargs)
        NLL = NLL_MM(X0, X1, pars_hat, dist = dist)
    elif set(dist) in [{'Bernoulli','Gaussian'}]:
        pars_hat = fit_BGM(X0, X1, K = K, dist = dist, **kwargs)
        NLL = NLL_BGM(X0, X1, pars_hat, dist = dist)
    elif set(dist) in [{'Bernoulli','Network'}]:
        pars_hat = fit_MM(X0, X1, K = K, dist = dist, **kwargs)
        NLL = NLL_MM(X0, X1, pars_hat, dist = dist)
    elif set(dist) in [{'Bernoulli'}]:
        pars_hat = fit_MM(X0, X1, K = K, dist = dist, **kwargs)
        NLL = NLL_MM(X0, X1, pars_hat, dist = dist)
    if show == True:
        print(NLL)
    # transfer learning by penalized NLL
    for lbd in list_lbd:
        Z_hat_lbd = TL_given_NLL(NLL = NLL, lbd = lbd)
        list_Z_hat.append(Z_hat_lbd)
    Z_hat_inf = DP_given_NLL(NLL)
    list_Z_hat.append(Z_hat_inf)
    return [pars_hat, list_Z_hat]


def TL_demo(X0, X1, Z, K, dist, list_lbd, show = False, **kwargs):
    [paras_hat, list_Z_hat] = TL_candidates(X0, X1, K, dist, list_lbd, show, **kwargs)

    errs, AMIs = [], []
    for j in range(len(list_lbd)+1):
        try:
            err_lbd = Hamming_aligned(Z, list_Z_hat[j])
        except:
            print(Z,list_Z_hat[j])
        AMI_lbd = adjusted_mutual_info_score(Z, list_Z_hat[j])
        errs.append(err_lbd)
        AMIs.append(AMI_lbd)
    
    return [paras_hat, errs, AMIs]


############################################################
# generate data

def generate_labels(n, paras):
    sizes = np.random.multinomial(n, paras['weights'], size = n)[0]
    ans = np.zeros(n).astype(int)
    tmp = 0
    for (k, n_k) in enumerate(sizes):
        ans[tmp : (tmp + n_k)] = k
        tmp += n_k
    return ans

def generate_source_labels(Z_0, epsilon):
    n = len(Z_0)
    Z = np.copy(Z_0)
    # Flip a proportion of labels in Z_0 according to epsilon
    flip_indices = np.random.choice(n, size=int(epsilon * n), replace=False)
    Z[flip_indices] = 1-Z_0[flip_indices]
    return Z

def generate_SBM_given_labels(Z, paras):
    A = np.random.binomial(1, p = np.triu( paras['probabilities'][Z][:, Z] , 1 ) )
    A = A + A.T
    return A


def generate_GMM_given_labels(Z, paras):
    K = np.max(Z) + 1  # number of clusters
    # Determine if it's one-dimensional or multivariate
    if np.isscalar(paras['means'][0]):
        # One-dimensional case
        X = np.zeros(len(Z))
    else:
        # Multivariate case
        X = np.zeros((len(Z), paras['means'][0].size))
    
    for k in range(K):
        idx_k = (Z == k)
        n_k = np.sum(idx_k)
        mu_k, Sigma_k = paras['means'][k], paras['covariances'][k]
        
        if np.isscalar(mu_k):
            # One-dimensional case
            X[idx_k] = np.random.normal(loc=mu_k, scale=np.sqrt(Sigma_k), size=n_k)
        else:
            # Multivariate case
            X[idx_k] = np.random.multivariate_normal(mean=mu_k, cov=Sigma_k, size=n_k)
    
    return X

def generate_LCM_given_labels(Z, paras):
    n = len(Z)
    p = paras['Theta'].shape[0]
    X = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            X[i, j] = np.random.binomial(1, paras['Theta'][j, Z[i]]) 
    return X

# bootstrap

def bootstrap(dist, paras, list_lbd, B_bootstrap = 500, seed = 0, refit = False, **kwargs):
    K = len(paras['X0']['weights'])
    M = len(list_lbd)
    diff = np.zeros((M, B_bootstrap))
    np.random.seed(seed)
    if dist == ['Gaussian','Gaussian']:
        for b in range(B_bootstrap):
        # directly use estimated labels on the target data, rather than generate random labels
            X0_b = generate_GMM_given_labels( paras['X0']['labels'], paras['X0'] )
            X1_b = generate_GMM_given_labels( paras['X0']['labels'], paras['X1'] )
            paras_b = paras
        # compute negative log-likelihood functions
            NLL_b = NLL_2GMM(X0_b, X1_b, paras_b)
            Z_infty_b = DP_given_NLL(NLL = NLL_b)
            for (j, lbd) in enumerate(list_lbd):
                Z_lbd_b = TL_given_NLL(NLL = NLL_b, lbd = lbd)
                diff[j, b] = np.sum(Z_lbd_b != Z_infty_b)
    elif dist == ['Network','Gaussian']:
        for b in range(B_bootstrap):
        # directly use estimated labels on the target data, rather than generate random labels
            A_b = generate_SBM_given_labels( paras['X0']['labels'], paras['X0'] )
            X_b = generate_GMM_given_labels( paras['X0']['labels'], paras['X1'] )
            paras_b = paras
            if refit: # refit models to estimate parameters
                paras_b = fit_CSBM(A_b, X_b, K = K, dist=dist, **kwargs)
        # compute negative log-likelihood functions
            NLL_b = NLL_CSBM(A_b, X_b, paras_b, dist)
            Z_infty_b = DP_given_NLL(NLL = NLL_b)
            for (j, lbd) in enumerate(list_lbd):
                Z_lbd_b = TL_given_NLL(NLL = NLL_b, lbd = lbd)
                diff[j, b] = np.sum(Z_lbd_b != Z_infty_b)
    elif dist == ['Gaussian','Network']:
        for b in range(B_bootstrap):
        # directly use estimated labels on the target data, rather than generate random labels
            A_b = generate_SBM_given_labels( paras['X0']['labels'], paras['X1'] )
            X_b = generate_GMM_given_labels( paras['X0']['labels'], paras['X0'] )
            paras_b = paras
            if refit: # refit models to estimate parameters
                paras_b = fit_CSBM(X_b, A_b, K = K, dist=dist, **kwargs)
        # compute negative log-likelihood functions
            NLL_b = NLL_CSBM(X_b, A_b, paras_b, dist)
            Z_infty_b = DP_given_NLL(NLL = NLL_b)
            for (j, lbd) in enumerate(list_lbd):
                Z_lbd_b = TL_given_NLL(NLL = NLL_b, lbd = lbd)
                diff[j, b] = np.sum(Z_lbd_b != Z_infty_b)
    elif dist == ['Bernoulli','Gaussian']:
        for b in range(B_bootstrap):
        # directly use estimated labels on the target data, rather than generate random labels
            X0_b = generate_LCM_given_labels( paras['X0']['labels'], paras['X0'] )
            X1_b = generate_GMM_given_labels( paras['X0']['labels'], paras['X1'] )
            paras_b = paras
        # compute negative log-likelihood functions
            NLL_b = NLL_BGM(X0_b, X1_b, paras_b, dist)
            Z_infty_b = DP_given_NLL(NLL = NLL_b)
            for (j, lbd) in enumerate(list_lbd):
                Z_lbd_b = TL_given_NLL(NLL = NLL_b, lbd = lbd)
                diff[j, b] = np.sum(Z_lbd_b != Z_infty_b)
    elif dist == ['Gaussian','Bernoulli']:
        for b in range(B_bootstrap):
        # directly use estimated labels on the target data, rather than generate random labels
            X0_b = generate_GMM_given_labels( paras['X0']['labels'], paras['X0'] )
            X1_b = generate_LCM_given_labels( paras['X0']['labels'], paras['X1'] )
            paras_b = paras
        # compute negative log-likelihood functions
            NLL_b = NLL_BGM(X0_b, X1_b, paras_b, dist)
            Z_infty_b = DP_given_NLL(NLL = NLL_b)
            for (j, lbd) in enumerate(list_lbd):
                Z_lbd_b = TL_given_NLL(NLL = NLL_b, lbd = lbd)
                diff[j, b] = np.sum(Z_lbd_b != Z_infty_b)
    elif dist == ['Bernoulli','Bernoulli']:
        for b in range(B_bootstrap):
        # directly use estimated labels on the target data, rather than generate random labels
            X0_b = generate_LCM_given_labels( paras['X0']['labels'], paras['X0'] )
            X1_b = generate_LCM_given_labels( paras['X0']['labels'], paras['X1'] )
            paras_b = paras
        # compute negative log-likelihood functions
            NLL_b = NLL_MM(X0_b, X1_b, paras_b, dist)
            Z_infty_b = DP_given_NLL(NLL = NLL_b)
            for (j, lbd) in enumerate(list_lbd):
                Z_lbd_b = TL_given_NLL(NLL = NLL_b, lbd = lbd)
                diff[j, b] = np.sum(Z_lbd_b != Z_infty_b)
  
    return diff


############################################################
# Adaptation

# error estimation
def estimate_error(X0, X1, K, dist, list_lbd, B_bootstrap = 10, list_q = [0.1], seed = 1000, refit = False, **kwargs):
    # Adaptation via Goldenshluger-Lepski
    [paras_hat, list_Z_hat] = TL_candidates(X0, X1, K, dist, list_lbd, **kwargs)
    Q = len(list_q)
    
    # psi_hat via bootstrap
    diff = bootstrap(dist = dist, paras = paras_hat, list_lbd = list_lbd, B_bootstrap = B_bootstrap, seed = seed, refit = refit, **kwargs)
    M = len(list_lbd)
    psi_hat = np.zeros((Q, M))
    for (k, q) in enumerate(list_q):
        for j in range(M):
            psi_hat[k, j] = np.quantile(diff[j], q)

    # phi_hat: bias estimation
    phi_hat = np.zeros((Q, M))
    for (k, q) in enumerate(list_q):
        for j in range(1, M):
            tmp = [max(0, np.sum(list_Z_hat[j] != list_Z_hat[_]) - psi_hat[k, _]) for _ in range(j)]
            # tmp = [max(0, np.sum(list_Z_hat[j] != list_Z_hat[_]) - psi_hat[k, _] - psi_hat[k, j]) for _ in range(j)]
            phi_hat[k, j] = max(tmp)
                
    return [phi_hat +  psi_hat, phi_hat, psi_hat]


############################################################

##### Generate samples #####
def generate_GMM_samples(K, n, mean, covariances, epsilon):
    parGMM = dict()
    parGMM['weights'] = np.array([1/K] * K)
    parGMM['means'] = np.array([mean, -mean])
    parGMM['covariances'] = np.array([covariances] * K)
    Z0 = generate_labels(n,parGMM)
    Z1 = generate_source_labels(Z0,epsilon)
    X0 = generate_GMM_given_labels(Z0,parGMM)
    X1 = generate_GMM_given_labels(Z1,parGMM)
    return [X0, X1, Z0, Z1, parGMM]


def generate_CSBM_samples(K, n, p, q, mean, covariances,  epsilon):
    parSBM = dict()
    parSBM['weights'] = np.array([1/K] * K)
    parSBM['probabilities'] =  np.full((K, K), q)
    np.fill_diagonal(parSBM['probabilities'], p)
    parGMM = dict()
    parGMM['weights'] = np.array([1/K] * K)
    parGMM['means'] = np.array([mean, -mean])
    parGMM['covariances'] = np.array([covariances] * K)
    Z0 = generate_labels(n,parSBM)
    Z1 = generate_source_labels(Z0,epsilon)
    X0 = generate_SBM_given_labels(Z0,parSBM)
    X1 = generate_GMM_given_labels(Z1,parGMM)
    return [X0, X1, Z0, Z1, parSBM, parGMM]

def generate_BGMM_samples(K, n, Theta, mean, covariances,  epsilon):
    parLCM = {'weights': np.array([1/K] * K), 'Theta':Theta}
    parGMM = {'weights': np.array([1/K] * K), 'means': np.array([mean, -mean]), 'covariances': np.array([covariances] * K)}
    Z0 = generate_labels(n,parLCM)
    Z1 = generate_source_labels(Z0,epsilon)
    X0 = generate_LCM_given_labels(Z0,parLCM)
    X1 = generate_GMM_given_labels(Z1,parGMM)
    return [X0, X1, Z0, Z1, parLCM, parGMM]


def generate_LCM_samples(K, n, Theta,  epsilon):
    parLCM = {'weights': np.array([1/K] * K), 'Theta':Theta}
    Z0 = generate_labels(n,parLCM)
    Z1 = generate_source_labels(Z0,epsilon)
    X0 = generate_LCM_given_labels(Z0,parLCM)
    X1 = generate_LCM_given_labels(Z1,parLCM)
    return [X0, X1, Z0, Z1, parLCM]


def one_sim(K, n, dist, mean, covariances, epsilon, B_bootstrap, list_q):
    [X0, X1, Z0, Z1, parGMM] = generate_GMM_samples(K,n, mean, covariances, epsilon)
    list_lbd = 0.2 * np.arange(20)
    # performance of all candidates
    [paras_hat, errs, AMIs] = TL_demo(X0, X1, Z0, K, dist, list_lbd)
   
    # adaptive clustering
    seed = 1000
    start = time.time()
    errs_hat = estimate_error(X0, X1, K = K, dist = dist, list_lbd = list_lbd, B_bootstrap = B_bootstrap, list_q = list_q, seed = seed, refit = False, num_refinement = 10, matrix = 'adjacency', regularizer = 'null')
    duration = time.time() - start
    err_mat = np.zeros(len(list_q)+2)
    err_mat[-2] = errs[0]
    err_mat[-1] = errs[-1]
    list_selected_idx = np.argmin(errs_hat, axis = 1)
    for (i, q) in enumerate(list_q):
        err_mat[i] = errs[list_selected_idx[i]]
    err_mat *= 1/n

    return err_mat


def ATC_err_lbd(K, n, dist, pars, epsilon, B_bootstrap, list_q, list_lbd, **kwargs):
    if dist==['Gaussian', 'Gaussian']:
        [X0, X1, Z0, Z1, parGMM] = generate_GMM_samples(K,n, pars['mean'], pars['covariances'], epsilon)
    elif dist==['Network','Gaussian']:
        [X0, X1, Z0, Z1, parSBM, parGMM] = generate_CSBM_samples(K, n, pars['p'], pars['q'], pars['mean'], pars['covariances'],  epsilon)
    elif dist==['Gaussian','Network']:
        [X1, X0, Z1, Z0, parSBM, parGMM] = generate_CSBM_samples(K, n, pars['p'], pars['q'], pars['mean'], pars['covariances'],  epsilon)
    elif dist==['Bernoulli','Gaussian']:
        [X0, X1, Z0, Z1, parLCM, parGMM] = generate_BGMM_samples(K, n, pars['Theta'], pars['mean'], pars['covariances'], epsilon)
    elif dist==['Bernoulli','Bernoulli']:
        [X0, X1, Z0, Z1, parLCM] = generate_LCM_samples(K, n, pars['Theta'], epsilon)
    
    # performance of all candidates
    [paras_hat, errs, AMIs] = TL_demo(X0, X1, Z0, K, dist, list_lbd, **kwargs)
   
    # adaptive clustering
    start = time.time()
    [errs_hat,_,_] = estimate_error(X0, X1, K = K, dist = dist, list_lbd = list_lbd, B_bootstrap = B_bootstrap, list_q = list_q, seed = 1000,  **kwargs)
    duration = time.time() - start
    err_vec = np.zeros(len(list_q)+2)
    err_vec[-2] = errs[0]
    err_vec[-1] = errs[-1]
    list_selected_idx = np.argmin(errs_hat, axis = 1)
    # print(list_selected_idx, errs_hat)
    for (i, q) in enumerate(list_q):
        err_vec[i] = errs[list_selected_idx[i]]
    err_vec *= 1/n

    return err_vec

