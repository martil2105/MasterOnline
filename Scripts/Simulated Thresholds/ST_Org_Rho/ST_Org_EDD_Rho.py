import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from time import time
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
stream_length_null = 50000
sigma = 1
seed = 2023
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 100
B_bound = np.array([0.25, 1.75])
rhos = 0.7
thresholds = [99999]
scale = 0
# Load model
current_file = "NN_train_rho"
model_name = "n100N400m24l5" #samme modell
logdir = Path("tensorboard_logs", f"{current_file}_100") #trent på rho 0.7 data
print(logdir)

model_path = Path(logdir, model_name, "model.keras")
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path)
np.random.seed(seed)
tf.random.set_seed(seed)


# Batched version of detection function
def detect_change_in_stream_loc_batched(stream, model, window_length, threshold):
    num_windows = len(stream) - window_length + 1
    windows = np.array([stream[i:i+window_length] for i in range(num_windows)])
    windows = np.expand_dims(windows, axis=-1) 
    logits = model.predict(windows, verbose=0)
    probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1]
    detection_idx = np.argmax(probs > threshold)
    if probs[detection_idx] > threshold:
        detection_time = detection_idx + window_length
    else:
        detection_time = 0
    return detection_time, np.max(probs)
result_alt =  DataGenAlternative( #data med changepoint og autokorrelasjon
                N_sub=num_repeats,
                B=B_val,
                mu_L=mu_L,
                n=stream_length,
                ARcoef=rhos, #autokorrelasjon verdi
                tau_bound=tau_bound,
                B_bound=B_bound,
                ar_model="AR0",#autokorrelasjon ar0
                sigma=sigma
            )
data_alt = result_alt["data"]     

true_tau_alt = result_alt["tau_alt"] 
# Generate null data and compute threshold percentile
data = GenDataMeanAR(num_repeats, stream_length_null, cp=None, mu=(mu_L, mu_L), sigma=sigma, coef=rhos) #data med autokorrelasjon
num_streams = data.shape[0]
false_alarm_rates = [0.8,0.85, 0.90,0.95,0.99]
output_dir = Path("datasets")
threshold_filepath = output_dir / "normal_thresholds_rho_100.npz"
loaded_thresholds = np.load(threshold_filepath)
percentiles_nn = loaded_thresholds["percentiles_nn"]
percentiles_cusum = loaded_thresholds["percentiles_cusum"]
percentiles_logit_cusum = loaded_thresholds["percentiles_logit_cusum"]
# Estimate ARL (Average Run Length)
arl_nn = np.zeros((len(percentiles_nn),num_repeats))
arl_cusum = np.zeros((len(percentiles_cusum),num_repeats))
arl_logit_cusum = np.zeros((len(percentiles_logit_cusum),num_repeats))
for idx in range(len(percentiles_nn)): #looper gjennom alle percentilene
    current_threshold_nn = percentiles_nn[idx]
    current_threshold_cusum = percentiles_cusum[idx]
    current_threshold_logit_cusum = percentiles_logit_cusum[idx]
    print(f"idx: {idx} current_threshold_nn: {current_threshold_nn} current_threshold_cusum: {current_threshold_cusum} current_threshold_logit_cusum: {current_threshold_logit_cusum}")
    for i in range(num_repeats):
        dt, _ = detect_change_in_stream_loc_batched(data[i], model, window_length, current_threshold_nn) #finner detection tid
        if dt > 0: #hvis det er en detection, så er det feil
            arl_nn[idx,i] = dt #arl er detection tid
        else:
            arl_nn[idx,i] = stream_length_null #hvis det ikke er en detection, så er arl lik stream length
        dt_cusum, _ = li_cusum(data[i], window_length, current_threshold_cusum)
        if dt_cusum > 0:
            arl_cusum[idx,i] = dt_cusum
        else:
            arl_cusum[idx,i] = stream_length_null
        dt_logit_cusum, _ = detect_change_in_stream_batched_cusum(data[i], model, window_length, current_threshold_logit_cusum)
        if dt_logit_cusum > 0:
            arl_logit_cusum[idx,i] = dt_logit_cusum
        else:
            arl_logit_cusum[idx,i] = stream_length_null

arl_cusum = np.mean(arl_cusum,axis=1) #finner gjennomsnittet av arl hver percentil
arl_nn = np.mean(arl_nn,axis=1)
arl_logit_cusum= np.mean(arl_logit_cusum,axis=1)


print(f"arl_cusum: {arl_cusum}")
print(f"arl_nn: {arl_nn}")
print(f"arl_logit_cusum: {arl_logit_cusum}")
print(f"arl_cusum: {arl_cusum}")
detection_delay_nn = np.zeros(len(percentiles_nn))
detection_delay_cusum= np.zeros(len(percentiles_cusum))
detection_delay_logit_cusum = np.zeros(len(percentiles_logit_cusum))

fp_cusum = np.zeros(len(percentiles_cusum))
fn_cusum = np.zeros(len(percentiles_cusum))
fp_nn = np.zeros(len(percentiles_nn))
fn_nn = np.zeros(len(percentiles_nn))
fp_logit_cusum = np.zeros(len(percentiles_logit_cusum))
fn_logit_cusum = np.zeros(len(percentiles_logit_cusum))
num_models = 4

detections_count_nn = np.zeros(len(percentiles_nn)) # Count of CPs detected by all three
detections_count_cusum = np.zeros(len(percentiles_cusum))
detections_count_logit_cusum = np.zeros(len(percentiles_logit_cusum))

for p_idx in range(len(percentiles_nn)):
    print(f"p_idx: {p_idx}")
    current_threshold_nn = percentiles_nn[p_idx]
    current_threshold_cusum = percentiles_cusum[p_idx]
    current_threshold_logit_cusum = percentiles_logit_cusum[p_idx]
    for i in range(num_repeats):
        dt_nn, _ = detect_change_in_stream_loc_batched(data_alt[i], model, window_length, current_threshold_nn)
        dt_cusum, _ = li_cusum(data_alt[i], window_length, current_threshold_cusum)
        dt_logit_cusum, _ = detect_change_in_stream_batched_cusum(data_alt[i], model, window_length, current_threshold_logit_cusum)
        delay_nn, delay_count_nn, fps_nn, fns_nn = find_detection_delay(stream_length, true_tau_alt[i], dt_nn)
        delay_cusum, delay_count_cusum, fps_cusum, fns_cusum = find_detection_delay(stream_length, true_tau_alt[i], dt_cusum)
        delay_logit_cusum, delay_count_logit_cusum, fps_logit_cusum, fns_logit_cusum = find_detection_delay(stream_length, true_tau_alt[i], dt_logit_cusum)
        detection_delay_nn[p_idx] += delay_nn
        detections_count_nn[p_idx] += delay_count_nn
        fp_nn[p_idx] += fps_nn
        fn_nn[p_idx] += fns_nn
        # cusum
        detection_delay_cusum[p_idx] += delay_cusum
        detections_count_cusum[p_idx] += delay_count_cusum
        fp_cusum[p_idx] += fps_cusum
        fn_cusum[p_idx] += fns_cusum
        # logit cusum
        detection_delay_logit_cusum[p_idx] += delay_logit_cusum
        detections_count_logit_cusum[p_idx] += delay_count_logit_cusum
        fp_logit_cusum[p_idx] += fps_logit_cusum
        fn_logit_cusum[p_idx] += fns_logit_cusum
        
    detection_delay_nn[p_idx] /= detections_count_nn[p_idx]
    detection_delay_cusum[p_idx] /= detections_count_cusum[p_idx]
    detection_delay_logit_cusum[p_idx] /= detections_count_logit_cusum[p_idx]

end_time = time()
print(f"Time taken: {end_time - begin_time} seconds")
plt.figure(figsize=(10, 8))
# Plot 1: Average Detection Delay
plt.subplot(2, 2, 1)
plt.plot(false_alarm_rates, detection_delay_nn, 'o-', linewidth=1.5, markersize=3)
plt.plot(false_alarm_rates, detection_delay_cusum, 'o-', linewidth=1.5, markersize=3, color='red')
plt.plot(false_alarm_rates, detection_delay_logit_cusum, 'o-', linewidth=1.5, markersize=3, color='green')  
plt.xlim(0.8,1.0)
plt.legend(['DD NN', 'DD CUSUM', 'DD Logit CUSUM'])
plt.xlabel('Percentile Threshold rho edd')
plt.ylabel('Average Detection Delay')
plt.title('Detection Delay vs Threshold')
plt.grid(True)

# Plot 2: False Positives
plt.subplot(2, 2, 2)
plt.plot(false_alarm_rates, fp_nn, 'o-', linewidth=1.5, markersize=3, color='blue')
plt.plot(false_alarm_rates, fn_nn, 'o-', linewidth=1.5, markersize=3, color='red')
plt.plot(false_alarm_rates, fp_cusum, 'o-', linewidth=1.5, markersize=3, color='green')
plt.plot(false_alarm_rates, fn_cusum, 'o-', linewidth=1.5, markersize=3, color='yellow')
plt.plot(false_alarm_rates, fp_logit_cusum, 'o-', linewidth=1.5, markersize=3, color='purple')
plt.plot(false_alarm_rates, fn_logit_cusum, 'o-', linewidth=1.5, markersize=3, color='orange')
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('Number of False Positives')
plt.title('False Positives vs Threshold')
plt.legend(['FP NN', 'FN NN', 'FP CUSUM', 'FN CUSUM', 'FP Logit CUSUM', 'FN Logit CUSUM'])
plt.grid(True)

# Plot 3: Average Run Length
plt.subplot2grid((2, 2), (1, 0), colspan=2)
plt.plot(false_alarm_rates, arl_nn, 'o-', linewidth=1.5, markersize=3, color='blue')
plt.plot(false_alarm_rates, arl_cusum, 'o-', linewidth=1.5, markersize=3, color='red')
plt.plot(false_alarm_rates, arl_logit_cusum, 'o-', linewidth=1.5, markersize=3, color='green')
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('Average Run Length')
plt.title('ARL vs Threshold')
plt.grid(True)
plt.legend(['ARL', 'ARL CUSUM', 'ARL Logit CUSUM'])
plt.savefig(f"Figures/ST_Org_EDD_Rho.png")
plt.tight_layout()
#plt.show()
