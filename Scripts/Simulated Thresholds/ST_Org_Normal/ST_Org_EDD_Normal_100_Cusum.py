import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import cProfile
import pstats
profiler = cProfile.Profile()
profiler.enable()
begin_time = time()

# Project path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
sys.path.append(project_root)

from autocpd.utils import *
from sklearn.utils import shuffle

# Parameters
window_length = 100
num_repeats = 1000
stream_length = 500
sigma = 1
seed = 2023
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 100
B_bound = np.array([0.25, 1.75])
rhos = 0
thresholds = [99999]
# Load model
current_file = "NN_train_normal"
model_name = "n100N400m24l5" #modell navn fra tensorboard, "vindu,antall serier trent på, bredde på NN, layers"
logdir = Path("tensorboard_logs", f"{current_file}") 
model_path = Path(logdir, model_name, "model.keras") #mappen
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path) #modellen

np.random.seed(seed)
tf.random.set_seed(seed)

# Batched version of detection function
def detect_change_in_stream_loc_batched(stream, model, window_length, threshold):
    num_windows = len(stream) - window_length + 1 #antall vinduer
    windows = np.array([stream[i:i+window_length] for i in range(num_windows)]) #har alle vinduene i en array for raskere inferens, men dette er akkurat det samme som en while loop
    windows = np.expand_dims(windows, axis=-1)  #slik at det passer NN
    logits = model.predict(windows, verbose=0) #finner en prediction for hvert av vinduene
    probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1] #sannynlighet for at det finnes et changepoint i vinduet
    detection_idx = np.argmax(probs > threshold) #finner den første changepointen, hvis det ikke er noen over threshold returneres 0
    if probs[detection_idx] > threshold: #hvis sannsynligheten er større enn threshold
        detection_time = detection_idx + window_length #changepoint er siste punkt i vinduet
    else:
        detection_time = 0 #returnerer 0
    return detection_time, np.max(probs) #detection tid og den største sannsynlighete for alle vinduene

def ComputeCUSUMM(x):
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
    mean_left = 0
    mean_right = np.mean(x[1:])
    a = np.repeat(0.0, n - 1)
    a[0,] = np.sqrt((n - 1) / n) * (mean_left - mean_right)
    for i in range(1, n - 1):
        mean_right = mean_right + (mean_right - x[i]) / (n - i - 1)
        a[i,] = np.sqrt((n - i - 1) * (i + 1) / n) * ( - mean_right)

    return a


def new_li_cusum(stream, window_length, threshold):
    num_windows = len(stream) - window_length + 1
    detection_times = 0
    max_cusum_scores = []
    for i in range(num_windows):
        window_data = stream[i:i+window_length]
        cusum_stats = ComputeCUSUMM(window_data)
        max_stat = np.max(np.abs(cusum_stats))
    
        max_cusum_scores.append(max_stat)
        
        if max_stat > threshold:
            detection_times = i+window_length
            break
    return detection_times, np.max(max_cusum_scores)
result_alt =  DataGenAlternative(
                N_sub=num_repeats,
                B=B_val,
                mu_L=mu_L,
                n=stream_length,
                ARcoef=rhos,
                tau_bound=tau_bound,
                B_bound=B_bound,
                ar_model="Gaussian",
                sigma=sigma) #Data med changepoint
data_alt = result_alt["data"] #tidseriene    
true_tau_alt = result_alt["tau_alt"] #changepointene


# Generate null data and compute threshold percentile
data_null = GenDataMean(num_repeats, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma) #null data
num_streams = data_null.shape[0]
max_probabilities = []
max_probabilities_cusum = []
max_probabilities_logit_cusum = []
false_alarm_rates = [0.8,0.85, 0.90,0.95,0.99] #forskjellige nivåer
output_dir = Path("datasets")
threshold_filepath = output_dir / "normal_thresholds.npz"
loaded_thresholds = np.load(threshold_filepath)
percentiles_nn = loaded_thresholds["percentiles"]
percentiles_cusum = loaded_thresholds["percentiles_cusum"]
percentiles_logit_cusum = loaded_thresholds["percentiles_logit_cusum"]

# Estimate ARL (Average Run Length)
arl = np.zeros(len(percentiles_nn)) #array for å finne ARL for ulike percentiler
arl_cusum = np.zeros(len(percentiles_cusum))
arl_logit_cusum = np.zeros(len(percentiles_logit_cusum))
data = GenDataMean(num_repeats, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma) #generer null data
num_streams = data.shape[0] #antall serier = repeats

for idx in range(len(percentiles_nn)):
    print(f"Processing threshold {idx+1} of {len(percentiles_nn)}")
    current_threshold_nn = percentiles_nn[idx]
    current_threshold_cusum = percentiles_cusum[idx]
    current_threshold_logit_cusum = percentiles_logit_cusum[idx]
    for i in range(num_streams): #looper gjennom alle serier
        if i % 100 == 0:
            print(f"Processing stream {i} of {num_streams}")
        dt_nn, _ = detect_change_in_stream_loc_batched(data[i], model, window_length, current_threshold_nn) #finner detection tid
        dt_cusum, _ = li_cusum(data[i], window_length, current_threshold_cusum)
        dt_logit_cusum, _ = detect_change_in_stream_batched_cusum(data[i], model, window_length, current_threshold_logit_cusum)
        if dt_nn > 0: #hvis det er en detection, så er det feil
            arl[idx] += dt_nn #arl er detection tid
        else:
            arl[idx] += stream_length #hvis det ikke er en detection, så er arl lik stream length   
        if dt_cusum > 0:
            arl_cusum[idx] += dt_cusum
        else:
            arl_cusum[idx] += stream_length
        if dt_logit_cusum > 0:
            arl_logit_cusum[idx] += dt_logit_cusum
        else:
            arl_logit_cusum[idx] += stream_length
arl_cusum = arl_cusum / num_streams
arl = arl / num_streams
arl_logit_cusum = arl_logit_cusum / num_streams
print(arl)
print(arl_cusum)
print(arl_logit_cusum)


detection_delay_nn = np.zeros(len(percentiles_nn))
detection_delay_cusum = np.zeros(len(percentiles_cusum))
detection_delay_logit_cusum = np.zeros(len(percentiles_logit_cusum))
fps_cusum = np.zeros(len(percentiles_cusum))
fns_cusum = np.zeros(len(percentiles_cusum))
fps_nn = np.zeros(len(percentiles_nn))
fns_nn = np.zeros(len(percentiles_nn))
fps_logit_cusum = np.zeros(len(percentiles_logit_cusum))
fns_logit_cusum = np.zeros(len(percentiles_logit_cusum))
delay_counts_nn = np.zeros(len(percentiles_nn))
delay_counts_cusum = np.zeros(len(percentiles_cusum))
delay_counts_logit_cusum = np.zeros(len(percentiles_logit_cusum))
for idx in range(len(percentiles_nn)):
    print(f"Processing threshold {idx+1} of {len(percentiles_nn)}")
    current_threshold_nn = percentiles_nn[idx]
    current_threshold_cusum = percentiles_cusum[idx]
    current_threshold_logit_cusum = percentiles_logit_cusum[idx]
    for i in range(num_repeats):
        dt_nn, _ = detect_change_in_stream_loc_batched(data_alt[i], model, window_length, current_threshold_nn)
        dt_cusum, _ = li_cusum(data_alt[i], window_length, current_threshold_cusum)
        dt_logit_cusum, _ = detect_change_in_stream_batched_cusum(data_alt[i], model, window_length, current_threshold_logit_cusum)
        delay_nn, delay_count_nn, fp_nn, fn_nn = find_detection_delay(stream_length, true_tau_alt[i], dt_nn)
        delay_cusum, delay_count_cusum, fp_cusum, fn_cusum = find_detection_delay(stream_length, true_tau_alt[i], dt_cusum)
        delay_logit_cusum, delay_count_logit_cusum, fp_logit_cusum, fn_logit_cusum = find_detection_delay(stream_length, true_tau_alt[i], dt_logit_cusum)
        detection_delay_nn[idx] += delay_nn
        delay_counts_nn[idx] += delay_count_nn
        detection_delay_cusum[idx] += delay_cusum
        delay_counts_cusum[idx] += delay_count_cusum
        detection_delay_logit_cusum[idx] += delay_logit_cusum
        delay_counts_logit_cusum[idx] += delay_count_logit_cusum
        fps_nn[idx] += fp_nn
        fns_nn[idx] += fn_nn
        fps_cusum[idx] += fp_cusum
        fns_cusum[idx] += fn_cusum
        fps_logit_cusum[idx] += fp_logit_cusum
        fns_logit_cusum[idx] += fn_logit_cusum
        
    detection_delay_nn[idx] = detection_delay_nn[idx] / delay_counts_nn[idx]
    detection_delay_cusum[idx] = detection_delay_cusum[idx] / delay_counts_cusum[idx]
    detection_delay_logit_cusum[idx] = detection_delay_logit_cusum[idx] / delay_counts_logit_cusum[idx]
plt.figure(figsize=(10, 8))
# Plot 1: Average Detection Delay
plt.subplot(2, 2, 1)
plt.plot(false_alarm_rates, detection_delay_nn, 'o-', linewidth=1.5, markersize=3, color='blue')
plt.plot(false_alarm_rates, detection_delay_cusum, 'o-', linewidth=1.5, markersize=3, color='red')
plt.plot(false_alarm_rates, detection_delay_logit_cusum, 'o-', linewidth=1.5, markersize=3, color='green')
plt.xlim(0.8,1.0)
plt.legend(['DD NN', 'DD CUSUM', 'DD Logit CUSUM'])
plt.xlabel('Percentile Threshold Normal')
plt.ylabel('Average Detection Delay')
plt.title('Detection Delay vs Threshold')
plt.grid(True)

# Plot 2: False Positives
plt.subplot(2, 2, 2)
plt.plot(false_alarm_rates, fps_nn, 'o-', linewidth=1.5, markersize=3, color='blue')
plt.plot(false_alarm_rates, fns_nn, 'o-', linewidth=1.5, markersize=3, color='yellow')
plt.plot(false_alarm_rates, fps_cusum, 'o-', linewidth=1.5, markersize=3, color='red')
plt.plot(false_alarm_rates, fns_cusum, 'o-', linewidth=1.5, markersize=3, color='orange')
plt.plot(false_alarm_rates, fps_logit_cusum, 'o-', linewidth=1.5, markersize=3, color='green')
plt.plot(false_alarm_rates, fns_logit_cusum, 'o-', linewidth=1.5, markersize=3, color='purple')
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('Number of False Positives')
plt.title('False Positives vs Threshold')
plt.legend(['FP NN', 'FN NN', 'FP CUSUM', 'FN CUSUM', 'FP Logit CUSUM', 'FN Logit CUSUM'])
plt.grid(True)

# Plot 3: Average Run Length
plt.subplot2grid((2, 2), (1, 0), colspan=2)
plt.plot(false_alarm_rates, arl, 'o-', linewidth=1.5, markersize=3, color='blue')
plt.plot(false_alarm_rates, arl_cusum, 'o-', linewidth=1.5, markersize=3, color='red')
plt.plot(false_alarm_rates, arl_logit_cusum, 'o-', linewidth=1.5, markersize=3, color='green')
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('Average Run Length')
plt.title('ARL vs Threshold')
plt.grid(True)
plt.legend(['ARL NN', 'ARL CUSUM', 'ARL Logit CUSUM'])
plt.savefig(f"Figures/ST_Org_EDD_Normal_100_Cusum.png")
plt.tight_layout()
#plt.show()
