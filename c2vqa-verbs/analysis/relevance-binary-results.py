import pandas as pd
import numpy as np
import os
from sklearn.metrics import *

results_directory = '/sb-personal/cvqa/results/c2vqa-verbs-results-final'

stats = []

for f in os.listdir(results_directory):
	if not f.endswith('test_results.csv'):
		continue

	print f
	parts = f.split('-')
	model = parts[0]
	trial = parts[1]
	path = os.path.join(results_directory, f)
	df = pd.read_csv(path)

	stat = {'model':model, 'trial':trial}

	if 'y_predict_relevance' in df:
		df['y_predict_relevance'] = df['y_predict_relevance'].apply(lambda x: 0 if x == "[False]" else 1)
	else:
		df['y_predict_relevance'] = df['y_predict'].apply(lambda x: 0 if x.startswith("no ") else 1)

	y_true = np.asarray(df['relevant'].tolist())
	y_predict = np.asarray(df['y_predict_relevance'].tolist())
	stat['accuracy'] = accuracy_score(y_true, y_predict)

	stats.append(stat)

stats_df = pd.DataFrame(stats)
# print stats_df

print stats_df.groupby(['model']).mean()

# finalStats = []
# for m in models:
# 	print m
# 	modelDf = allDf[allDf['model'] == m]
# 	row = modelDf.mean().to_dict()
# 	row['model'] = m
# 	finalStats.append(row)

# statsDf = pd.DataFrame(finalStats)

# statsDf = statsDf.sort_values(['norm_accuracy'])

# cols = ['norm_accuracy']
# for _,m in statsDf.iterrows():
# 	# print m
# 	items = [m[c] * 100 for c in cols]
# 	items = ["%.2f" % i for i in items]
# 	print " & ".join([modelNames[m['model']]] + items) + " \\\\"