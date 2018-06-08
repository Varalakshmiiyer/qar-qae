import pandas as pd
import numpy as np
import os
from sklearn.metrics import *

results_directory = '/sb-personal/cvqa/results/check/'
df = pd.read_csv(results_directory + 'qcatt-0-test_results.csv')
print df[df['y_predict'] != df['y_true']].sample(10)[['question','answer','caption','relevant','y_true','y_predict']]