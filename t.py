import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def linear_assignment(scores):
    scores = scores.numpy()
    scores = np.exp(-1*scores)
    rids0,cids0 = linear_sum_assignment(scores)
    x,y = scores.shape
    p0 = np.full(x, 1)
    indices0 = np.full(x, -1)
    for r,c in zip(rids0,cids0):
        p0[r] = scores[r,c]
        indices0[r] = c
    rids1,cids1 = linear_sum_assignment(scores.T)
    p1 = np.full(y, 1)
    indices1 = np.full(y, -1)
    for r,c in zip(rids1,cids1):
        p1[r] = scores.T[r,c]
        indices1[r] = c
    p0 = -np.log(p0)
    indices0 = torch.tensor(indices0)
    p1 = -np.log(p1)
    indices1 = torch.tensor(indices1)
    return p0, indices0, p1, indices1

def linear_assignment_torch(scores):
    scores = scores.numpy()
    scores = np.exp(-1*scores)
    f = open("a.txt", 'a')
    f.write("\n")
    f.write(scores)
    f.close()
    rows,cols = linear_sum_assignment(scores)
    chosen_scores = torch.sparse_coo_tensor(torch.tensor(np.array([rows,cols])), torch.ones(rows.shape))
    chosen_scores_dense = chosen_scores.to_dense()
    selected_entries = (chosen_scores_dense*scores)
    return selected_entries

if __name__ == '__main__':
    sc = np.matrix([[1,2,3,4],
                    [9,4,7,6],
                    [5,6,4,3]])

    score = torch.tensor(sc)
    #p0, ind0, p1, ind1 = linear_assignment(score)
    sco = linear_assignment_torch(score)
    sco = torch.stack([sco], dim = 0)
    score = torch.stack([score], dim = 0)
    sco = torch.nn.Parameter(sco)
    bin_score = torch.nn.Parameter(torch.tensor(1.))
    scores = log_optimal_transport(score,bin_score,100)
    max0, max1 = sco.max(2), sco.max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = sco.new_tensor(0)
    mscores0 = torch.where(mutual0, - max0.values.log(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > 0)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
    print(sco)
    print('---------------------')
    print(mutual0)
    print('---------------------')
    print(mutual1)
    print('---------------------')
    print(max0)
    print('---------------------')
    print(max1)
    print('---------------------')
    print(mscores0)
    print('---------------------')
    print(mscores1)
    print('---------------------')
    print(indices0)
    print('---------------------')
    print(indices1)


    max0_, max1_ = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0_, indices1_ = max0_.indices, max1_.indices
    mutual0_ = arange_like(indices0_, 1)[None] == indices1_.gather(1, indices0_)
    mutual1_ = arange_like(indices1_, 1)[None] == indices0_.gather(1, indices1_)
    zero_ = scores.new_tensor(0)
    mscores0_ = torch.where(mutual0_, max0_.values.exp(), zero_)
    mscores1_ = torch.where(mutual1_, mscores0_.gather(1, indices1_), zero_)
    valid0_ = mutual0_ & (mscores0_ > 0.2)
    valid1_ = mutual1_ & valid0_.gather(1, indices1_)
    indices0_ = torch.where(valid0_, indices0_, indices0_.new_tensor(-1))
    indices1_ = torch.where(valid1_, indices1_, indices1_.new_tensor(-1))
    print('---------------------')
    print(mutual0_)
    print('---------------------')
    print(mutual1_)
    print('---------------------')
    print(max0_)
    print('---------------------')
    print(max1_)
    print('---------------------')
    print(mscores0_)
    print('---------------------')
    print(mscores1_)
    print('---------------------')
    print(indices0_)
    print('---------------------')
    print(indices1_)