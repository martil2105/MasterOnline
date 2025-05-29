import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from time import time
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
scale = 0.5 #cauchy skalering
# Load model
current_file = "NN_train_cauchy_100"
model_name = "n100N400m24l5" #modell navn fra tensorboard, "vindu,antall serier trent på, bredde på NN, layers"
logdir = Path("tensorboard_logs", f"{current_file}") 
model_path = Path(logdir, model_name, "model.keras") #mappen
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path) #modellen

np.random.seed(seed)
tf.random.set_seed(seed)

data_null = GenDataMeanARH(num_repeats, stream_length, cp=None, mu=(mu_L, mu_L), coef=rhos, scale=scale) #nulldata med cauchy, gå til utils for mer info
num_streams = data_null.shape[0]
max_probabilities = []
max_probabilities_cusum = []
max_probabilities_logit_cusum = []
dt, max_prob = detect_change_in_stream_loc_batched(data_null[0], model, window_length, thresholds[0])
print(f"dt: {dt}, max_prob: {max_prob}")
for i in range(num_streams):
    if i % 10000 == 0:
        print(f"i: {i}")
    dt, max_prob = detect_change_in_stream_loc_batched(data_null[i], model, window_length, thresholds[0])
    max_probabilities.append(max_prob)
    dt_cusum, max_prob_cusum = li_cusum(data_null[i], window_length, thresholds[0])
    max_probabilities_cusum.append(max_prob_cusum)
    dt_logit_cusum, max_prob_logit_cusum = detect_change_in_stream_batched_cusum(data_null[i], model, window_length, thresholds[0])
    max_probabilities_logit_cusum.append(max_prob_logit_cusum)
false_alarm_rates = [0.80, 0.85, 0.90,0.95,0.99]
#regular
percentile_80 = np.percentile(max_probabilities, 80)
percentile_85 = np.percentile(max_probabilities, 85)
percentile_90 = np.percentile(max_probabilities, 90)
percentile_95 = np.percentile(max_probabilities, 95)
percentile_99 = np.percentile(max_probabilities, 99)
#logit cusum
percentile_80_logit_cusum = np.percentile(max_probabilities_logit_cusum, 80)
percentile_85_logit_cusum = np.percentile(max_probabilities_logit_cusum, 85)
percentile_90_logit_cusum = np.percentile(max_probabilities_logit_cusum, 90)
percentile_95_logit_cusum = np.percentile(max_probabilities_logit_cusum, 95)
percentile_99_logit_cusum = np.percentile(max_probabilities_logit_cusum, 99)
#cusum
percentile_80_cusum = np.percentile(max_probabilities_cusum, 80)
percentile_85_cusum = np.percentile(max_probabilities_cusum, 85)
percentile_90_cusum = np.percentile(max_probabilities_cusum, 90)
percentile_95_cusum = np.percentile(max_probabilities_cusum, 95)
percentile_99_cusum = np.percentile(max_probabilities_cusum, 99)
print(f"Repeats: {num_streams}, 95th percentile: {percentile_95}")
percentiles_nn = [percentile_80,percentile_85,percentile_90, percentile_95, percentile_99]
percentiles_cusum = [percentile_80_cusum,percentile_85_cusum,percentile_90_cusum, percentile_95_cusum, percentile_99_cusum]
percentiles_logit_cusum = [percentile_80_logit_cusum,percentile_85_logit_cusum,percentile_90_logit_cusum, percentile_95_logit_cusum, percentile_99_logit_cusum]
output_dir = Path("datasets")
threshold_filepath = output_dir / "normal_thresholds_cauchy_100.npz"
np.savez_compressed(threshold_filepath, percentiles_nn=percentiles_nn, percentiles_cusum=percentiles_cusum, percentiles_logit_cusum=percentiles_logit_cusum)
print(f"Thresholds saved to: {threshold_filepath}")
loaded_thresholds = np.load(threshold_filepath)
print(loaded_thresholds.keys())
