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
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(project_root)

from autocpd.utils import *
from sklearn.utils import shuffle

# Parameters
window_length = 100
num_repeats = 3000
stream_length = 500
stream_length_null = 500
sigma = 1
seed = 2023
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 100
B_bound = np.array([0.25, 1.75])
rhos = 0.7
thresholds = [99999]

# Load model
current_file = "NN_train_rho"
model_name = "n100N400m24l5" #modell navn fra tensorboard, "vindu,antall serier trent på, bredde på NN, layers"
logdir = Path("tensorboard_logs", f"{current_file}_100") 
model_path = Path(logdir, model_name, "model.keras") #mappen
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path) #modellen

np.random.seed(seed)
tf.random.set_seed(seed)


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
            ) #Data med changepoint
data_alt = result_alt["data"] #tidseriene    
true_tau_alt = result_alt["tau_alt"] #changepointene

# Generate null data and compute threshold percentile
false_alarm_rates = [0.8,0.85, 0.90,0.95,0.99]
output_dir = Path("datasets")
threshold_filepath = output_dir / "thresholds_rho_100.npz"
loaded_thresholds = np.load(threshold_filepath)
percentiles_nn = loaded_thresholds["percentiles_nn"]
percentiles_cusum = loaded_thresholds["percentiles_cusum"]
percentiles_logit = loaded_thresholds["percentiles_logit"]
percentiles_smart_cusum = loaded_thresholds["percentiles_smart_cusum"]
# Estimate ARL (Average Run Length)
arl_nn = np.zeros((len(percentiles_nn),num_repeats)) #array for å finne ARL for ulike percentiler
arl_cusum = np.zeros((len(percentiles_cusum),num_repeats))
arl_logit = np.zeros((len(percentiles_logit),num_repeats))
arl_smart_cusum = np.zeros((len(percentiles_smart_cusum),num_repeats))
data = GenDataMeanAR(num_repeats, stream_length_null, cp=None, mu=(mu_L, mu_L), sigma=sigma, coef=rhos)  #generer null data
num_streams = data.shape[0] #antall serier = repeats
for idx in range(len(percentiles_nn)): #looper gjennom alle percentilene
    current_threshold_nn = percentiles_nn[idx]
    current_threshold_cusum = percentiles_cusum[idx]
    current_threshold_logit = percentiles_logit[idx]
    current_threshold_smart_cusum = percentiles_smart_cusum[idx]
    print(f"idx: {idx} current_threshold_nn: {current_threshold_nn} current_threshold_cusum: {current_threshold_cusum} current_threshold_logit: {current_threshold_logit} current_threshold_smart_cusum: {current_threshold_smart_cusum}")
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
        dt_logit, _ = detect_change_in_stream_batched_cusum(data[i], model, window_length, current_threshold_logit)
        if dt_logit > 0:
            arl_logit[idx,i] = dt_logit
        else:
            arl_logit[idx,i] = stream_length_null
        dt_smart_cusum, _ = smart_li_cusum(data[i], window_length, current_threshold_smart_cusum)
        if dt_smart_cusum > 0:
            arl_smart_cusum[idx,i] = dt_smart_cusum
        else:
            arl_smart_cusum[idx,i] = stream_length_null

arl_cusum = np.mean(arl_cusum,axis=1) #finner gjennomsnittet av arl hver percentil
arl_nn = np.mean(arl_nn,axis=1)
arl_logit= np.mean(arl_logit,axis=1)
arl_smart_cusum= np.mean(arl_smart_cusum,axis=1)


detection_delay_nn = np.zeros((len(percentiles_nn),num_repeats))
detection_delay_cusum= np.zeros((len(percentiles_cusum),num_repeats))
detection_delay_logit = np.zeros((len(percentiles_logit),num_repeats))
detection_delay_smart_cusum = np.zeros((len(percentiles_smart_cusum),num_repeats))
fp_cusum = np.zeros(len(percentiles_cusum))
fn_cusum = np.zeros(len(percentiles_cusum))
fp_nn = np.zeros(len(percentiles_nn))
fn_nn = np.zeros(len(percentiles_nn))
fp_logit = np.zeros(len(percentiles_logit))
fn_logit = np.zeros(len(percentiles_logit))
fp_smart_cusum = np.zeros(len(percentiles_smart_cusum))
fn_smart_cusum = np.zeros(len(percentiles_smart_cusum))
num_models = 4
detection_times_nn = np.zeros((num_repeats,len(percentiles_nn)))
detection_times_cusum = np.zeros((num_repeats,len(percentiles_cusum)))
detection_times_logit = np.zeros((num_repeats,len(percentiles_logit)))
detection_times_smart_cusum = np.zeros((num_repeats,len(percentiles_smart_cusum)))
delay_nn = np.zeros((num_repeats,len(percentiles_nn)))
delay_cusum = np.zeros((num_repeats,len(percentiles_cusum)))
delay_logit = np.zeros((num_repeats,len(percentiles_logit)))
delay_smart_cusum = np.zeros((num_repeats,len(percentiles_smart_cusum)))
common_detections_count = np.zeros(len(percentiles_nn)) # Count of CPs detected by all three
fair_delay_nn = np.zeros(len(percentiles_nn))
fair_delay_cusum = np.zeros(len(percentiles_cusum))
fair_delay_logit = np.zeros(len(percentiles_logit))
fair_delay_smart_cusum = np.zeros(len(percentiles_smart_cusum))

for p_idx in range(len(percentiles_nn)):
    print(f"p_idx: {p_idx}")
    current_threshold_nn = percentiles_nn[p_idx]
    current_threshold_cusum = percentiles_cusum[p_idx]
    current_threshold_logit = percentiles_logit[p_idx]
    current_threshold_smart_cusum = percentiles_smart_cusum[p_idx]
    for i in range(num_repeats):
        dt_nn, _ = detect_change_in_stream_loc_batched(data_alt[i], model, window_length, current_threshold_nn)
        dt_cusum, _ = li_cusum(data_alt[i], window_length, current_threshold_cusum)
        dt_logit, _ = detect_change_in_stream_batched_cusum(data_alt[i], model, window_length, current_threshold_logit)
        dt_smart_cusum, _ = smart_li_cusum(data_alt[i], window_length, current_threshold_smart_cusum)
        delay_nn, delay_count_nn, fps_nn, fns_nn = find_detection_delay(stream_length, true_tau_alt[i], dt_nn)
        delay_cusum, delay_count_cusum, fps_cusum, fns_cusum = find_detection_delay(stream_length, true_tau_alt[i], dt_cusum)
        delay_logit, delay_count_logit, fps_logit, fns_logit = find_detection_delay(stream_length, true_tau_alt[i], dt_logit)
        delay_smart_cusum, delay_count_smart_cusum, fps_smart_cusum, fns_smart_cusum = find_detection_delay(stream_length, true_tau_alt[i], dt_smart_cusum)
        fp_nn[p_idx] += fps_nn
        fn_nn[p_idx] += fns_nn
        fp_cusum[p_idx] += fps_cusum
        fn_cusum[p_idx] += fns_cusum
        fp_logit[p_idx] += fps_logit
        fn_logit[p_idx] += fns_logit
        fp_smart_cusum[p_idx] += fps_smart_cusum
        fn_smart_cusum[p_idx] += fns_smart_cusum
        if delay_count_nn > 0 and delay_count_cusum > 0 and delay_count_logit > 0 and delay_count_smart_cusum > 0:
            common_detections_count[p_idx] += 1
            fair_delay_nn[p_idx] += delay_nn
            fair_delay_cusum[p_idx] += delay_cusum
            fair_delay_logit[p_idx] += delay_logit
            fair_delay_smart_cusum[p_idx] += delay_smart_cusum
        
    print(f"fair delay nn: {fair_delay_nn[p_idx]}")
    print(f"fair delay cusum: {fair_delay_cusum[p_idx]}")
    print(f"fair delay logit cusum: {fair_delay_logit[p_idx]}")
    print(f"fair delay smart cusum: {fair_delay_smart_cusum[p_idx]}")

    fair_delay_nn[p_idx] /= common_detections_count[p_idx]
    fair_delay_cusum[p_idx] /= common_detections_count[p_idx]
    fair_delay_logit[p_idx] /= common_detections_count[p_idx]
    fair_delay_smart_cusum[p_idx] /= common_detections_count[p_idx]
    print(f"fair delay nn: {fair_delay_nn[p_idx]}")
    print(f"fair delay cusum: {fair_delay_cusum[p_idx]}")
    print(f"fair delay logit cusum: {fair_delay_logit[p_idx]}")
    print(f"fair delay smart cusum: {fair_delay_smart_cusum[p_idx]}")

print(f"Common detections: {common_detections_count}")
print(f"Fair delay nn: {fair_delay_nn}")
print(f"Fair delay cusum: {fair_delay_cusum}")
print(f"Fair delay logit cusum: {fair_delay_logit}")
print(f"Fair delay smart cusum: {fair_delay_smart_cusum}")

plt.figure(figsize=(10, 10))
# Plot 1: Average Detection Delay 
plt.subplot(2, 2, 1)
plt.plot(false_alarm_rates, fair_delay_nn, 'o-', linewidth=1.5, markersize=3, color='blue') #nn er blå
plt.plot(false_alarm_rates, fair_delay_cusum, 'o-', linewidth=1.5, markersize=3, color='red') #cusum er rød
plt.plot(false_alarm_rates, fair_delay_logit, 'o-', linewidth=1.5, markersize=3, color='green') #logit er grønn
plt.plot(false_alarm_rates, fair_delay_smart_cusum, 'o-', linewidth=1.5, markersize=3, color='purple') #smart cusum er lilla
plt.xlabel('Percentile Threshold Normal')
plt.ylabel('Average Detection Delay')
plt.title('Detection Delay vs Threshold')
plt.grid(True)
plt.legend(['NN', 'CUSUM', 'Logit', 'Smart CUSUM'])

# Plot 2: False Positives
plt.subplot(2, 2, 2)
plt.plot(false_alarm_rates, fp_nn/num_repeats, 'o-', linewidth=1.5, markersize=3, color='blue')
plt.plot(false_alarm_rates, fp_cusum/num_repeats, 'o-', linewidth=1.5, markersize=3, color='red')
plt.plot(false_alarm_rates, fp_logit/num_repeats, 'o-', linewidth=1.5, markersize=3, color='green')
plt.plot(false_alarm_rates, fp_smart_cusum/num_repeats, 'o-', linewidth=1.5, markersize=3, color='purple')
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('False Positive Rate')
plt.title('False Positive Rate vs Threshold')
plt.legend(['NN', 'CUSUM', 'Logit', 'Smart CUSUM'])
plt.grid(True)
#Plot 3: False Negatives
plt.subplot(2, 2, 3)
plt.plot(false_alarm_rates, fn_nn/num_repeats, 'o-', linewidth=1.5, markersize=3, color='blue')
plt.plot(false_alarm_rates, fn_cusum/num_repeats, 'o-', linewidth=1.5, markersize=3, color='red')
plt.plot(false_alarm_rates, fn_logit/num_repeats, 'o-', linewidth=1.5, markersize=3, color='green')
plt.plot(false_alarm_rates, fn_smart_cusum/num_repeats, 'o-', linewidth=1.5, markersize=3, color='purple')
plt.xlim(0.8,1.0)
plt.legend(['NN', 'CUSUM', 'Logit', 'Smart CUSUM'])
plt.xlabel('Percentile Threshold')
plt.ylabel('False Negative Rate')
plt.title('False Negative Rate vs Threshold')
plt.grid(True)
# Plot 3: Average Run Length
plt.plot(2,2,4)
plt.plot(false_alarm_rates, arl_nn, 'o-', linewidth=1.5, markersize=3, color='blue')
plt.plot(false_alarm_rates, arl_cusum, 'o-', linewidth=1.5, markersize=3, color='red')
plt.plot(false_alarm_rates, arl_logit, 'o-', linewidth=1.5, markersize=3, color='green')
plt.plot(false_alarm_rates, arl_smart_cusum, 'o-', linewidth=1.5, markersize=3, color='purple')
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('Average Run Length')
plt.title('ARL vs Threshold')
plt.grid(True)
plt.legend(['NN', 'CUSUM', 'Logit', "Smart CUSUM"])
plt.savefig(f"Figures/ST_Same_EDD_Rho_3000run.png")
plt.tight_layout()
#plt.show() 

"""Bruk 100 000 MC samples for å velge thresholds
Måten å regne Detection Delay: ta gjennomsnitt over ALLE sanne changepoints. De blir som ikke blir oppdaget, sett straff på 500. 
Legg inn en ekstra variant av CUSUM som VET at pre-change mean er 0
Kjør alle simuleringer på N = 1000 samples. 
Plott falske positive og falske negative hver for seg
 """