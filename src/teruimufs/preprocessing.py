import numpy as np

def construct_UIMdata(data, per):
    view_num = len(data)
    ratio = {
        1: [1], 2: [0.5, 1.5], 3: [0.5, 1, 1.5],
        4: [0.25, 0.75, 1.25, 1.75], 5: [0.25, 0.75, 1, 1.25, 1.75],
        6: [0.25, 0.5, 1, 1, 1.5, 1.75],
    }
    if view_num not in ratio: raise ValueError(f"Unsupported view_num: {view_num}")
    instance_num = data[0].shape[0]
    perDel = 0.01 * per * np.array(ratio[view_num])
    result, missing_num, X_missing, zero_indices, one_indices = [], [], [], [], []
    
    for view_idx in range(view_num):
        m_num = int(np.fix(instance_num * perDel[view_idx]))
        missing_num.append(m_num)
        oz_indices = np.ones(instance_num, dtype=bool)
        z_idx = np.random.permutation(instance_num)[:m_num]
        oz_indices[z_idx] = False
        o_idx = np.where(oz_indices)[0]
        zero_indices.append(z_idx)
        one_indices.append(o_idx)
        
        res = data[view_idx].copy()
        x_miss = res[o_idx, :]
        res[z_idx, :] = 0
        
        # ترانهاده برای مطابقت با فرمت متلب (ویژگی‌ها × نمونه‌ها)
        result.append(res.T)
        X_missing.append(x_miss.T)
        
    return result, missing_num, X_missing, zero_indices, one_indices