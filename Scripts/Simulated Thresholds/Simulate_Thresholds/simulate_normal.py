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
num_repeats = 100000
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




# Generate null data and compute threshold percentile
data_null = GenDataMean(num_repeats, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma) #null data
num_streams = data_null.shape[0]
max_probabilities = []
max_probabilities_cusum = []
max_probabilities_smart_cusum = []
max_probabilities_logit = []

for i in range(num_streams):
    if i % 10 == 0:
        print(f"Processing stream {i+1} of {num_streams}, percent done: {i/num_streams*100}%")
    dt, max_prob = detect_change_in_stream_loc_batched(data_null[i], model, window_length, thresholds[0]) #med urealistisk høy threshold for å finne maksen under null
    max_probabilities.append(max_prob) #legger til en høyeste sannsynlighet
    dt_cusum, max_prob_cusum = li_cusum(data_null[i], window_length, thresholds[0]) #li cusum for alle vinduene med urealistisk høy threshold for å finne maksen under null
    max_probabilities_cusum.append(max_prob_cusum) #legger til i cusum listen
    dt_logit_cusum, max_prob_logit = detect_change_in_stream_batched_cusum(data_null[i], model, window_length, thresholds[0]) #logit differanse cusum
    max_probabilities_logit.append(max_prob_logit) #legges i riktig liste
    dt_smart_cusum, max_prob_smart_cusum = smart_li_cusum(data_null[i], window_length, thresholds[0]) #smart li cusum for alle vinduene med urealistisk høy threshold for å finne maksen under null
    max_probabilities_smart_cusum.append(max_prob_smart_cusum) #legges i riktig liste
false_alarm_rates = [0.8,0.85, 0.90,0.95,0.99] #forskjellige nivåer
#regular
percentile_80 = np.percentile(max_probabilities, 80) #persentil for ulike nivåer
percentile_85 = np.percentile(max_probabilities, 85)
percentile_90 = np.percentile(max_probabilities, 90)
percentile_95 = np.percentile(max_probabilities, 95)
percentile_99 = np.percentile(max_probabilities, 99)
#logit cusum
percentile_80_logit = np.percentile(max_probabilities_logit, 80)
percentile_85_logit = np.percentile(max_probabilities_logit, 85)
percentile_90_logit = np.percentile(max_probabilities_logit, 90)
percentile_95_logit = np.percentile(max_probabilities_logit, 95)
percentile_99_logit = np.percentile(max_probabilities_logit, 99)
#cusum
percentile_80_cusum = np.percentile(max_probabilities_cusum, 80)
percentile_85_cusum = np.percentile(max_probabilities_cusum, 85)
percentile_90_cusum = np.percentile(max_probabilities_cusum, 90)
percentile_95_cusum = np.percentile(max_probabilities_cusum, 95)
percentile_99_cusum = np.percentile(max_probabilities_cusum, 99)
#smart cusum
percentile_80_smart_cusum = np.percentile(max_probabilities_smart_cusum, 80)
percentile_85_smart_cusum = np.percentile(max_probabilities_smart_cusum, 85)
percentile_90_smart_cusum = np.percentile(max_probabilities_smart_cusum, 90)
percentile_95_smart_cusum = np.percentile(max_probabilities_smart_cusum, 95)
percentile_99_smart_cusum = np.percentile(max_probabilities_smart_cusum, 99)
print(f"Repeats: {num_streams}, 95th percentile: {percentile_95}")
percentiles_nn = [percentile_80,percentile_85,percentile_90, percentile_95, percentile_99] #percentiler slik at vi kan loope gjennom dem
percentiles_cusum = [percentile_80_cusum,percentile_85_cusum,percentile_90_cusum, percentile_95_cusum, percentile_99_cusum]
percentiles_logit = [percentile_80_logit,percentile_85_logit,percentile_90_logit, percentile_95_logit, percentile_99_logit]
percentiles_smart_cusum = [percentile_80_smart_cusum,percentile_85_smart_cusum,percentile_90_smart_cusum, percentile_95_smart_cusum, percentile_99_smart_cusum]
print(percentiles_nn)
print(percentiles_cusum)
print(percentiles_logit)
output_dir = Path("datasets")
threshold_filepath = output_dir / "normal_thresholds.npz"
np.savez_compressed(threshold_filepath, percentiles_nn=percentiles_nn, percentiles_cusum=percentiles_cusum, percentiles_logit=percentiles_logit, percentiles_smart_cusum=percentiles_smart_cusum)
print(f"Thresholds saved to: {threshold_filepath}")
loaded_thresholds = np.load(threshold_filepath)
print(f"Files in loaded_thresholds: {loaded_thresholds.files}")
print(f"Loaded thresholds: {loaded_thresholds}")
