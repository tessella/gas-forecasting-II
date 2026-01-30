import numpy as np
import iisignature
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sktime.dists_kernels import SignatureKernel

# weights for regime similarity:
def weights_gas_robust(horizon, window, temperature, data_x, time_indices, level):
    t = len(data_x)
    anchor_end = t - horizon
    
    anchor_raw_x = data_x[anchor_end - window : anchor_end]
    anchor_time = time_indices[anchor_end - window : anchor_end]
    anchor_z = np.concatenate([anchor_time, anchor_raw_x], axis=1)
    
    sig_kernel = SignatureKernel(level=level)
    
    if temperature < 1e-5:
        return np.ones(t - horizon - window + 1) / (t - horizon - window + 1)

    dists = []
    for tau in range(window, t - horizon + 1):
        hist_raw_x = data_x[tau - window : tau]
        hist_time = time_indices[tau - window : tau]
        hist_z = np.concatenate([hist_time, hist_raw_x], axis=1)
        
        k_xx = sig_kernel(anchor_z.reshape(1, window, -1), anchor_z.reshape(1, window, -1))[0,0]
        k_yy = sig_kernel(hist_z.reshape(1, window, -1), hist_z.reshape(1, window, -1))[0,0]
        k_xy = sig_kernel(anchor_z.reshape(1, window, -1), hist_z.reshape(1, window, -1))[0,0]
        dist = k_xx - 2*k_xy + k_yy
        dists.append(dist)
    
    dists = np.array(dists)
    
    # NOTE: numerically stable scaling, normal can ruin results
    med_dist = np.median(dists)
    if med_dist == 0: med_dist = 1.0
    scaled_dists = dists / med_dist
    
    weights = np.exp(-temperature * scaled_dists)
    weights /= np.sum(weights)
    
    return weights

def ridge_gas_refined(data_x, data_y, dates, max_horizon, window, sig_depth, 
                       temperature, level, cv_folds=5):
    
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    dates = pd.to_datetime(dates)

    # NOTE: hardcoding these features internally may cause redundancy. 
    # I'll maybe change this later

    day_of_year = dates.dt.dayofyear
    doy_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    
    temp_weekend = data_x[:, 0] * data_x[:, 2]
    wind_chill = data_x[:, 0] * data_x[:, 1]
    
    data_x_augmented = np.column_stack([
        data_x, 
        doy_sin, 
        doy_cos, 
        temp_weekend, 
        wind_chill
    ])
    
    scaler_x = StandardScaler()
    data_x_scaled = scaler_x.fit_transform(data_x_augmented) #no lookahead bias as weather data considered stationary 
    
    data_y = data_y.reshape(-1, 1)
    data_y_log = np.log(data_y)

    time_indices = np.arange(len(dates)).reshape(-1, 1)
    time_indices = time_indices / np.max(time_indices)

    t = len(data_x_scaled)
    forecasts = np.zeros(max_horizon)
    
    for delta_t in range(1, max_horizon + 1):
        # A: regime weights 
        w = weights_gas_robust(delta_t, window, temperature, data_x_scaled, time_indices, level)
        w = w * len(w) # for model weights, else too small
        
        # B: build features
        x_features = []
        y_targets = []
        
        for i, tau in enumerate(range(window, t - delta_t + 1)):
            # 1 - time-augmented features:
            path_x = data_x_scaled[tau - window : tau]
            path_time = time_indices[tau - window : tau]
            path_z = np.concatenate([path_time, path_x], axis=1)
            sig = iisignature.sig(path_z, sig_depth)
            # 2 - target:
            val_at_target = data_x_scaled[tau + delta_t - 1]
            # 3 - build X: 
            features = np.concatenate([val_at_target, sig])
            x_features.append(features)
            y_targets.append(data_y_log[tau + delta_t - 1, 0])
            
        x_features = np.array(x_features)
        y_targets = np.array(y_targets)
        
        # C: fit ridge
        scaler_feat = StandardScaler()
        x_final = scaler_feat.fit_transform(x_features)
        
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=cv_folds)
        ridge.fit(x_final, y_targets, sample_weight=w)
        
        # D: finally, predict
        curr_path_x = data_x_scaled[t - window : t]
        curr_time = time_indices[t - window : t]
        curr_path_z = np.concatenate([curr_time, curr_path_x], axis=1)
        curr_sig = iisignature.sig(curr_path_z, sig_depth)
        
        idx_target = t - 1 - (max_horizon - delta_t)
        curr_val_at_target = data_x_scaled[idx_target]
        
        curr_features = np.concatenate([curr_val_at_target, curr_sig])
        curr_features_scaled = scaler_feat.transform(curr_features.reshape(1, -1))
        
        pred_log = ridge.predict(curr_features_scaled)[0]
        forecasts[delta_t - 1] = np.exp(pred_log)
        
    return forecasts, None
