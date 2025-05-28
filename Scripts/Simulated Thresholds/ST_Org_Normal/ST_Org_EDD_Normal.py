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


# Generate null data and compute threshold percentile
false_alarm_rates = [0.8,0.85, 0.90,0.95,0.99]
output_dir = Path("datasets")
threshold_filepath = output_dir / "normal_thresholds.npz"
loaded_thresholds = np.load(threshold_filepath)
percentiles_nn = loaded_thresholds["percentiles"]
percentiles_cusum = loaded_thresholds["percentiles_cusum"]
percentiles_logit_cusum = loaded_thresholds["percentiles_logit_cusum"]

# Estimate ARL (Average Run Length)
arl_nn = np.zeros((len(percentiles_nn),num_repeats)) #array for å finne ARL for ulike percentiler
arl_cusum = np.zeros((len(percentiles_cusum),num_repeats))
arl_logit_cusum = np.zeros((len(percentiles_logit_cusum),num_repeats))
data = GenDataMean(num_repeats, stream_length_null, cp=None, mu=(mu_L, mu_L), sigma=sigma) #generer null data
num_streams = data.shape[0] #antall serier = repeats

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
print(arl_nn)
print(arl_cusum)
print(arl_logit_cusum)
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
        detection_delay_cusum[p_idx] += delay_cusum
        detections_count_cusum[p_idx] += delay_count_cusum
        fp_cusum[p_idx] += fps_cusum
        fn_cusum[p_idx] += fns_cusum
        detection_delay_logit_cusum[p_idx] += delay_logit_cusum
        detections_count_logit_cusum[p_idx] += delay_count_logit_cusum
        fp_logit_cusum[p_idx] += fps_logit_cusum
        fn_logit_cusum[p_idx] += fns_logit_cusum
    detection_delay_nn[p_idx] /= detections_count_nn[p_idx]
    detection_delay_cusum[p_idx] /= detections_count_cusum[p_idx]
    detection_delay_logit_cusum[p_idx] /= detections_count_logit_cusum[p_idx]



plt.figure(figsize=(10, 8))
# Plot 1: Average Detection Delay 
plt.subplot(2, 2, 1)
plt.plot(false_alarm_rates, detection_delay_nn, 'o-', linewidth=1.5, markersize=3, color='blue') #nn er blå
plt.plot(false_alarm_rates, detection_delay_cusum, 'o-', linewidth=1.5, markersize=3, color='red') #cusum er rød
plt.plot(false_alarm_rates, detection_delay_logit_cusum, 'o-', linewidth=1.5, markersize=3, color='green') #logit cusum er grønn
plt.xlim(0.8,1.0)
plt.legend(['DD NN', 'DD CUSUM', 'DD Logit CUSUM'])
plt.xlabel('Percentile Threshold Normal')
plt.ylabel('Average Detection Delay')
plt.title('Detection Delay vs Threshold')
plt.grid(True)

# Plot 2: False Positives
plt.subplot(2, 2, 2)
plt.plot(false_alarm_rates, fp_nn, 'o-', linewidth=1.5, markersize=3, color='blue')
plt.plot(false_alarm_rates, fn_nn, 'o-', linewidth=1.5, markersize=3, color='yellow')
plt.plot(false_alarm_rates, fp_cusum, 'o-', linewidth=1.5, markersize=3, color='red')
plt.plot(false_alarm_rates, fn_cusum, 'o-', linewidth=1.5, markersize=3, color='orange')
plt.plot(false_alarm_rates, fp_logit_cusum, 'o-', linewidth=1.5, markersize=3, color='green')
plt.plot(false_alarm_rates, fn_logit_cusum, 'o-', linewidth=1.5, markersize=3, color='purple')
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
plt.legend(['ARL NN', 'ARL CUSUM', 'ARL Logit CUSUM'])
plt.savefig(f"Figures/ST_Org_EDD_Normal.png")
plt.tight_layout()
#plt.show() 



"""
Ordered by: cumulative time
   List reduced from 3856 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    11000   19.677    0.002 1534.984    0.140 /Users/martin/Downloads/Skole/Master/MasterOnline/autocpd/utils.py:290(li_cusum)
 15702912 1372.049    0.000 1477.944    0.000 /Users/martin/Downloads/Skole/Master/MasterOnline/autocpd/utils.py:179(ComputeCUSUM)
  4433207  505.097    0.000  508.690    0.000 {built-in method tensorflow.python._pywrap_tfe.TFE_Py_FastPathExecute}
38642797/33581505    8.918    0.000  424.660    0.000 {built-in method builtins.next}
 12507746    1.402    0.000  319.779    0.000 {built-in method builtins.iter}
  1342000    0.260    0.000  319.574    0.000 /opt/anaconda3/lib/python3.12/site-packages/keras/src/backend/tensorflow/trainer.py:735(__next__)
  1342000    0.476    0.000  318.901    0.000 /opt/anaconda3/lib/python3.12/site-packages/keras/src/trainers/epoch_iterator.py:96(_enumerate_iterator)
    44000    0.156    0.000  318.378    0.007 /opt/anaconda3/lib/python3.12/site-packages/tensorflow/python/data/ops/dataset_ops.py:488(__iter__)
    44000    0.068    0.000  317.589    0.007 /opt/anaconda3/lib/python3.12/site-packages/tensorflow/python/data/ops/iterator_ops.py:670(__init__)
    44000    0.251    0.000  317.521    0.007 /opt/anaconda3/lib/python3.12/site-packages/tensorflow/python/data/ops/iterator_ops.py:713(_create_iterator)
    44000    0.044    0.000  314.719    0.007 /opt/anaconda3/lib/python3.12/site-packages/tensorflow/python/ops/gen_dataset_ops.py:3460(make_iterator)
  1320000    2.846    0.000  120.849    0.000 /opt/anaconda3/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py:1267(_call_flat)
  1320000    1.425    0.000  114.530    0.000 /opt/anaconda3/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py:214(call_preflattened)
  1320000   10.659    0.000  109.580    0.000 /opt/anaconda3/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py:219(call_flat)
  1452000    2.322    0.000  105.203    0.000 /opt/anaconda3/lib/python3.12/site-packages/tensorflow/core/function/polymorphism/function_cache.py:43(lookup)
"""