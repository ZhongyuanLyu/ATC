import numpy as np

def preprocess_lawyer(name_label = 'status', name_network = 'work', list_cov = ['years w/ firm'], remove_isolated = True, remove_office_3 = True):
    # Lawyer data
    # name_network = 'work', 'friend', 'adv'
    # e.g., [name_label, name_network, names_cov] = ['status', 'friend', ['years w/ firm']]

    list_names = ['seniority', 'status', 'gender', 'office', 'years w/ firm', 'age', 'practice', 'school']
    folder_dataset = 'LazegaLawyers/'
    path_network = folder_dataset + 'EL{}.dat'.format(name_network)
    path_cov = folder_dataset + 'ELattr.dat'

    X_0 = np.loadtxt(path_cov)
    A = np.loadtxt(path_network)
    A = (A + A.T > 0) + 0 # symmetrize
        
    # remove nodes
    idx_to_remove = []
    if remove_office_3:
        idx_to_remove += list(np.where(X_0[:, 3] == 3)[0]) # remove office 3
    if remove_isolated:
        deg = np.sum(A, axis=1)
        idx_to_remove += list(np.where(deg == 0)[0])

    A = np.delete(A, idx_to_remove, axis = 0)
    A = np.delete(A, idx_to_remove, axis = 1)
    X_0 = np.delete(X_0, idx_to_remove, axis=0)

    Z = X_0[:, list_names.index(name_label)].astype(int) - 1

    cov_idx = []
    for name in list_cov:
        cov_idx.append( list_names.index(name) )
    X = X_0[:, cov_idx].reshape(len(X_0), -1)

    return [A, X, Z]


def preprocess_politician():
    # Mexican political data    
    folder_dataset = 'Datasets/politicians/'
    edges = np.loadtxt(folder_dataset + 'network.txt')
    n = 35
    A = np.zeros((n, n)).astype(int)
    for edge in edges:
        A[int(edge[0]) - 1, int(edge[1]) - 1] = 1
        A[int(edge[1]) - 1, int(edge[0]) - 1] = 1
    
    X = np.loadtxt(folder_dataset + 'years.txt')
    X = X.reshape(len(X), -1)
    Z = np.loadtxt(folder_dataset + 'partition.txt').astype(int) - 1
    return [A, X, Z]

