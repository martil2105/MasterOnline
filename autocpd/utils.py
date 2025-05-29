import os
import posixpath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.stats import cauchy

def GenDataMean(N, n, cp, mu, sigma):
    """
    The function  generates the data for change in mean with Gaussian noise.
    When "cp" is None, it generates the data without change point.

    Parameters
    ----------
    N : int
        the sample size
    n : int
        the length of time series
    cp : int
        the change point, only 1 change point is accepted in this function.
    mu : float
        the piecewise mean
    sigma : float
        the standard deviation of Gaussian distribution

    Returns
    -------
    numpy array
        2D array with size (N, n)
    """
    if cp is None:
        data = np.random.normal(mu[0], sigma, (N, n))
    else:
        data1 = np.random.normal(mu[0], sigma, (N, cp))
        data2 = np.random.normal(mu[1], sigma, (N, n - cp))
        data = np.concatenate((data1, data2), axis=1)
    return data

def DataGenAlternative(
    N_sub,
    B,
    mu_L,
    n,
    B_bound,
    sigma,
    ARcoef=0.0,
    tau_bound=2,
    ar_model="Gaussian",
    scale=0.1,
    
):
    """
    This function genearates the simulation data from alternative model of change in mean.

    Parameters
    ----------
    N_sub : int
        The sample size of simulation data.
    B : float
        The signal-to-noise ratio of parameter space.
    mu_L : float
        The single at the left of change point.
    n : int
        The length of time series.
    B_bound : list, optional
        The upper and lower bound scalars of signal-to-noise.
    ARcoef : float, optional
        The autoregressive parameter of AR(1) model, by default 0.0
    tau_bound : int, optional
        The lower bound of change point, by default 2
    ar_model : str, optional
        The different models, by default 'Gaussian'. ar_model="AR0" means AR(1)
        noise with autoregressive parameter 'ARcoef'; ar_model="ARH" means
        Cauchy noise with scale parameter 'scale'; ar_model="ARrho" means AR(1)
        noise with random autoregressive parameter 'scale';
    scale : float, optional
        The scale parameter of Cauchy distribution, by default 0.1
    sigma : float, optional
        The standard variance of normal distribution, by default 1.0

    Returns
    -------
    dict
        data: size (N_sub,n);
        tau_alt: size (N_sub,); the change points
        mu_R: size (N_sub,); the single at the right of change point
    """
    tau_all = np.random.randint(low=tau_bound, high=n - tau_bound, size=N_sub)
    eta_all = tau_all / n
    mu_R_abs_lower = B / np.sqrt(eta_all * (1 - eta_all))
    # max_mu_R = np.max(mu_R_abs_lower)
    sign_all = np.random.choice([-1, 1], size=N_sub)
    mu_R_all = np.zeros((N_sub,))
    data = np.zeros((N_sub, n))
    for i in range(N_sub):
        mu_R = mu_L - sign_all[i] * np.random.uniform(
            low=B_bound[0] * mu_R_abs_lower[i],
            high=B_bound[1] * mu_R_abs_lower[i],
            size=1,
        )
        mu_R_all[i] = mu_R[0]
        mu = np.array([mu_L, mu_R[0]], dtype=np.float32)
        if ar_model == "Gaussian":
            data[i, :] = GenDataMean(1, n, cp=tau_all[i], mu=mu, sigma=sigma)
        elif ar_model == "AR0":
            data[i, :] = GenDataMeanAR(1, n, cp=tau_all[i], mu=mu, sigma=1, coef=ARcoef)
        elif ar_model == "ARH":
            data[i, :] = GenDataMeanARH(
                1, n, cp=tau_all[i], mu=mu, coef=ARcoef, scale=scale
            )
        elif ar_model == "ARrho":
            data[i, :] = GenDataMeanARrho(1, n, cp=tau_all[i], mu=mu, sigma=sigma)
        elif ar_model == "Variance":
            if isinstance(sigma, list) and len(sigma) == 2:
                sigma_L, sigma_R = sigma
            else:
                raise ValueError("For 'Variance' type, sigma must be a list of two values: [sigma_L, sigma_R]")
            data[i, :] = GenDataVariance(1, n, cp=tau_all[i], mu=mu_L, sigma=[sigma_L, sigma_R])
    return {"data": data, "tau_alt": tau_all, "mu_R_alt": mu_R_all}

def GenDataMeanAR(N, n, cp, mu, sigma, coef):
    """
    The function  generates the data for change in mean with AR(1) noise.
    When "cp" is None, it generates the data without change point.

    Parameters
    ----------
    N : int
        the sample size
    n : int
        the length of time series
    cp : int
        the change point, only 1 change point is accepted in this function.
    mu : float
        the piecewise mean
    sigma : float
        the standard deviation of Gaussian innovations in AR(1) noise
    coef : float scalar
        the coefficients of AR(1) model

    Returns
    -------
    numpy array
        2D array with size (N, n)
    """
    arparams = np.array([1, -coef])
    maparams = np.array([1])
    ar_process = ArmaProcess(arparams, maparams)
    if cp is None:
        data = mu[0] + np.transpose(
            ar_process.generate_sample(nsample=(n, N), scale=sigma)
        )
    else:
        noise = ar_process.generate_sample(nsample=(n, N), scale=sigma)
        signal = np.repeat(mu, (cp, n - cp))
        data = np.transpose(noise) + signal
    return data

def MaxCUSUM(x):
    """
    To return the maximum of CUSUM

    Parameters
    ----------
    x : vector
        the time series

    Returns
    -------
    scalar
        the maximum of CUSUM
    """
    y = np.abs(ComputeCUSUM(x))
    return np.max(y)

def ComputeCUSUM(x):
    """
    Compute the CUSUM statistics with O(n) time complexity

    Parameters
    ----------
    x : vector
        the time series

    Returns
    -------
    vector
        a: the CUSUM statistics vector.
    """
    n = len(x)
    mean_left = x[0]
    mean_right = np.mean(x[1:])
    a = np.repeat(0.0, n - 1)
    a[0,] = np.sqrt((n - 1) / n) * (mean_left - mean_right)
    for i in range(1, n - 1):
        mean_left = mean_left + (x[i] - mean_left) / (i + 1)
        mean_right = mean_right + (mean_right - x[i]) / (n - i - 1)
        a[i,] = np.sqrt((n - i - 1) * (i + 1) / n) * (mean_left - mean_right)

    return a

def detect_change_in_stream(stream, model, window_length,k, threshold ):

    num_windows = len(stream) - window_length + 1
    predictions = np.zeros(num_windows)
    consecutive_changes = 0
    detected_change = 0
    detected_change_windows = []
    for i in range(num_windows):
        window_data = stream[i: i + window_length]
        window_input = np.expand_dims(window_data, axis=0)
        logits = model.predict(window_input, verbose=0)
        probs = tf.nn.softmax(logits, axis=1)
        prob_change = probs[0,1]
        print(f"prob_change: {probs}")
        if prob_change > threshold:
            predictions[i] = 1
        else:
            predictions[i] = 0
        if predictions[i] == 1:
            #print(f"Change detected at window {i}")
            consecutive_changes += 1
        else: 
            consecutive_changes = 0
        if consecutive_changes >= k:
            detected_change = 1
            detected_change_windows.append(i)
            #print(f"Change detected at window {i}")
        
    return detected_change#, detected_change_windows
#print(DataGenAlternative(1,,mu_L,n=100,B_bound=B_bound,sigma=1,ar_model="Gaussian"))
def detect_change_in_stream_batched_cusum(stream, model, window_length, threshold):
    logits_difference = []
    num_windows = len(stream) - window_length + 1
    windows = np.array([stream[i:i+window_length] for i in range(num_windows)])
    windows = np.expand_dims(windows, axis=-1)

    logits = model.predict(windows, verbose=0)
    logits_diff = logits[:,1] - logits[:,0] #d_t 
    #print(logits_diff)
    cusum_scores = []
    detection_times = 0
    S = 0
    for i in range(len(logits_diff)):
        d_t = logits_diff[i]
        S = max(0, S + d_t)
        cusum_scores.append(S)
        #print(cusum_scores)
        if S > threshold:
            detection_times = i+window_length
            break
    return detection_times, np.max(cusum_scores)

def li_cusum(stream, window_length, threshold):
    num_windows = len(stream) - window_length + 1
    detection_times = 0
    max_cusum_scores = []
    for i in range(num_windows):
        window_data = stream[i:i+window_length]
        cusum_stats = ComputeCUSUM(window_data)
        max_stat = np.max(np.abs(cusum_stats))
    
        max_cusum_scores.append(max_stat)
        
        if max_stat > threshold:
            detection_times = i+window_length
            break
    return detection_times, np.max(max_cusum_scores)

def GenDataMeanARH(N, n, cp, mu, coef, scale):
    """
	The function  generates the data for change in mean with Cauchy noise with location parameter 0 and scale parameter 'scale'. When "cp" is None, it generates the data without change point.

	Parameters
	----------
	N : int
		the number of data
	n : int
		the number of variables
	cp : int
		the change point, only 1 change point is accepted in this function.
	mu : float
		the piecewise mean
	coef : float array
		the coefficients of AR(p) model
	scale : the scale parameter of Cauchy distribution
		the coefficients of AR(p) model

	Returns
	-------
	numpy array
		2D array with size (N, n)
	"""
	# initialize
    n1 = n + 30
    x_0 = np.ones((N,), dtype=np.float64)
	# eps_mat = np.random.standard_cauchy((N, n1))
    eps_mat = cauchy.rvs(loc=0, scale=scale, size=(N, n1))
    noise_mat = np.empty((N, n1))
    for i in range(n1):
        x_0 = coef * x_0 + eps_mat[:, i]
        noise_mat[:, i] = x_0

    if cp is None:
        data = mu[0] + noise_mat[:, -n:]
    else:
        signal = np.repeat(mu, (cp, n - cp))
        data = signal + noise_mat[:, -n:]

    return data

def detect_change_in_stream_loc_batched(stream, model, window_length, threshold):
    num_windows = len(stream) - window_length + 1 #antall vinduer
    windows = np.array([stream[i:i+window_length] for i in range(num_windows)]) #har alle vinduene i en array for raskere inferens, men dette er akkurat det samme som en while loop
    windows = np.expand_dims(windows, axis=-1)  #slik at det passer NN
    logits = model.predict(windows, verbose=0) #finner en prediction for hvert av vinduene
    probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1] #sannynlighet for at det finnes et changepoint i vinduet
    detection_idx = np.argmax(probs > threshold) #finner den første changepointen
    if probs[detection_idx] > threshold: 
        detection_time = detection_idx + window_length
    else:
        detection_time = 0
    return detection_time, np.max(probs)

def find_detection_delay(stream_length,change_point, detection_time):
    delay_count = 0
    fp = 0 
    fn = 0
    if detection_time > 0 and detection_time >= change_point:
        detection_delay = detection_time - change_point
        delay_count = 1
    if detection_time > 0 and detection_time < change_point:
        detection_delay = 0
        delay_count = 0
        fp = 1
    if detection_time == 0 and change_point > 0:
        detection_delay = stream_length
        delay_count = 1
        fn = 1
    return detection_delay, delay_count, fp, fn
    
