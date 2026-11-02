import numpy as np
from .graph import constructW, optSigma, Update_S, solve_delta
from .tensor_ops import wshrinkObj
from .optimization import solve_l1l2
from .initialization import init_H

def TERUIMUFS(X, XF, W, missing_num, class_num, maxiter, alpha, beta, tau, omega, pp):
    view_num = len(X)
    instance_num = X[0].shape[1]
    dim_num = [x.shape[0] for x in X]
    
    gamma = 10000
    sX = [instance_num, instance_num, view_num]
    mu1 = mu2 = mu3 = mu4 = 0.1
    mu_max = 1e10
    eta = 2
    epson = 1e-7
    
    lambda_ = np.ones(view_num) / view_num
    
    A = []
    for i in range(view_num):
        sigma = optSigma(X[i].T)
        Ai = constructW(X[i].T, k=5, WeightMode='HeatKernel', t=sigma)
        Ai = view_num * Ai / np.tile(np.sum(Ai, axis=1, keepdims=True), (1, Ai.shape[0]))
        A.append(Ai)
        
    AC = sum(lambda_[i] * A[i] for i in range(view_num))
    SS = (AC + AC.T) / 2
    L = np.diag(np.sum(SS, axis=1)) - SS
    
    H = init_H(XF.T, class_num, method='k_means')
    
    C = [np.eye(dim_num[i], class_num) for i in range(view_num)]
    B = [C[i].T @ X[i] @ H for i in range(view_num)]
    D = [np.eye(dim_num[i]) for i in range(view_num)]
    E = [np.zeros((dim_num[i], missing_num[i])) for i in range(view_num)]
    XN = [x.copy() for x in X]
    
    J1 = [np.zeros((dim_num[i], instance_num)) for i in range(view_num)]
    J2 = [np.zeros((instance_num, instance_num)) for i in range(view_num)]
    J3 = [np.zeros((instance_num, instance_num)) for i in range(view_num)]
    J4 = [np.zeros((dim_num[i], instance_num)) for i in range(view_num)]
    
    B1 = [np.zeros((dim_num[i], instance_num)) for i in range(view_num)]
    B2 = [np.zeros((instance_num, instance_num)) for i in range(view_num)]
    
    Z = [a.copy() for a in A]
    Y = [a.copy() for a in A]
    
    for iter_ in range(maxiter):
        lambda_ = solve_delta(AC, A)
        F8 = []
        for i in range(view_num):
            temp_W = XN[i] @ XN[i].T + tau * (X[i] @ L @ X[i].T) + alpha * D[i]
            # معادل temp_W \ ... در متلب
            C[i] = np.linalg.solve(temp_W, XN[i] @ H @ B[i].T)
            tempD = 0.5 * pp * (np.sqrt(np.sum(C[i]**2, axis=1) + epson))**(pp - 2)
            D[i] = np.diag(tempD)
            F8.append(C[i].T @ X[i])
            
        for i in range(view_num):
            SVD = C[i].T @ XN[i] @ H
            U, S, Vh = np.linalg.svd(SVD, full_matrices=False)
            B[i] = U @ Vh
            
        HN = np.maximum(H, 0)
        SVD = sum(XN[i].T @ C[i] @ B[i] for i in range(view_num)) + gamma * HN
        U, S, Vh = np.linalg.svd(SVD, full_matrices=False)
        H = U @ Vh
        
        AC = Update_S(A, F8, view_num, 1, tau, lambda_)
        SS = (AC + AC.T) / 2
        L = np.diag(np.sum(SS, axis=1)) - SS
        
        Ktemp = []
        for i in range(view_num):
            F1 = X[i] - XN[i] + J4[i]/mu4
            F2 = W[i] - W[i] @ Z[i]
            F3 = X[i] - X[i] @ Z[i] - B1[i] + J1[i]/mu1
            E_l = -mu4 * F1 @ W[i].T - mu1 * (F3 @ F2.T)
            m = missing_num[i]
            E_r = 2 * beta * np.ones((m, m)) + (-2 * beta + mu4) * np.eye(m) + mu1 * (F2 @ F2.T)
            E[i] = np.linalg.solve(E_r.T, E_l.T).T
            Ktemp.append(X[i] + E[i] @ W[i])
            
        for i in range(view_num):
            F4 = B[i] @ H.T
            F5 = X[i] + E[i] @ W[i] + J4[i]/mu4
            XN_l = 2 * C[i] @ C[i].T + mu4 * np.eye(dim_num[i])
            XN_r = 2 * C[i] @ F4 + mu4 * F5
            XN[i] = np.linalg.solve(XN_l, XN_r)
            
        H1H = np.vstack([-Ktemp[i] @ Z[i] + Ktemp[i] + J1[i]/mu1 for i in range(view_num)])
        B1B = solve_l1l2(H1H, 1/mu1)
        start = 0
        for i in range(view_num):
            d = dim_num[i]
            B1[i] = B1B[start:start+d, :]
            start += d
            
        H2H = np.vstack([Z[i] - A[i] + J2[i]/mu2 for i in range(view_num)])
        B2B = solve_l1l2(H2H, 1/mu2)
        start = 0
        for i in range(view_num):
            B2[i] = B2B[start:start+instance_num, :]
            start += instance_num
            
        for i in range(view_num):
            F6 = Z[i] - B2[i] + J2[i]/mu2
            F7 = -Y[i] + J3[i]/mu3
            A_temp = 2 * alpha * AC + mu2 * F6 - mu3 * F7
            A[i] = A_temp / (2 * alpha**2 + mu2 + mu3)
            
        for i in range(view_num):
            F9 = X[i] + E[i] @ W[i] - B1[i] + J1[i]/mu1
            F10 = -A[i] - B2[i] + J2[i]/mu2
            Z_left = mu1 * Ktemp[i].T @ Ktemp[i] + mu2 * np.eye(instance_num)
            Z_right = mu1 * Ktemp[i].T @ F9 - mu2 * F10
            Z[i] = np.linalg.solve(Z_left, Z_right)
            
        A_tensor = np.stack(A, axis=-1)
        J3_tensor = np.stack(J3, axis=-1)
        aT = A_tensor.ravel()
        j3T = J3_tensor.ravel()
        yT, _ = wshrinkObj(aT + 1/mu3 * j3T, 1/mu3, sX, 0, 3, omega)
        Y_tensor = yT.reshape(sX)
        for i in range(view_num): Y[i] = Y_tensor[:, :, i]
        
        for i in range(view_num):
            J1[i] += mu1 * (Ktemp[i] - Ktemp[i] @ Z[i] - B1[i])
            J2[i] += mu2 * (Z[i] - A[i] - B2[i])
            J3[i] += mu3 * (A[i] - Y[i])
            J4[i] += mu4 * (Ktemp[i] - XN[i])
            
        mu1 = min(mu_max, mu1 * eta)
        mu2 = min(mu_max, mu2 * eta)
        mu3 = min(mu_max, mu3 * eta)
        mu4 = min(mu_max, mu4 * eta)
        
    WW = np.vstack(C)
    # مرتب‌سازی نزولی معادل sort(..., 'descend')
    ranking = np.argsort(np.sum(WW**2, axis=1))[::-1] 
    return ranking