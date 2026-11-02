import numpy as np

def wshrinkObj(x, rho, sX, isWeight, mode, omega):
    X = x.reshape(sX)
    # معادل shiftdim(X, 1) در متلب
    if mode == 3:
        Y = np.moveaxis(X, 0, -1) 
    else:
        Y = X
        
    Yhat = np.fft.fft(Y, axis=-1)
    objV = 0.0
    
    if mode == 3: n3 = sX[0]
    elif mode == 1: n3 = sX[1]
    else: n3 = sX[2]
        
    is_even = (n3 % 2 == 0)
    endValue = n3 // 2 + 1
    
    for i in range(1, endValue + 1):
        idx = i - 1
        slice_i = Yhat[:, :, idx]
        U, S, Vh = np.linalg.svd(slice_i, full_matrices=False)
        
        if not isWeight:
            # ساخت tau هم‌اندازه S
            tau = np.zeros(len(S))
            for l in range(min(len(S), len(omega))):
                tau[l] = rho * omega[l]
            S_thresh = np.maximum(S - tau, 0)
        else:
            S_thresh = S 
            
        objV += np.sum(S_thresh)
        Yhat[:, :, idx] = U @ np.diag(S_thresh) @ Vh
        
        # حفظ تقارن مزدوج برای سیگنال‌های حقیقی
        if i > 1:
            conj_idx = n3 - i + 1
            if conj_idx < n3:
                Yhat[:, :, conj_idx] = np.conj(Yhat[:, :, idx])
                objV += np.sum(S_thresh)
                
    if is_even:
        idx = endValue
        if idx < n3:
            slice_i = Yhat[:, :, idx]
            U, S, Vh = np.linalg.svd(slice_i, full_matrices=False)
            if not isWeight:
                tau = rho * omega[:len(S)]
                S_thresh = np.maximum(S - tau, 0)
            else:
                S_thresh = S
            objV += np.sum(S_thresh)
            Yhat[:, :, idx] = U @ np.diag(S_thresh) @ Vh
            
    Y = np.fft.ifft(Yhat, axis=-1)
    
    if mode == 3:
        X_out = np.moveaxis(Y, -1, 0)
    else:
        X_out = Y
        
    return np.real(X_out).ravel(), objV