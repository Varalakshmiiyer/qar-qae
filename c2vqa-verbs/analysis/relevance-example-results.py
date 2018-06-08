import pandas as pd
import numpy as np
import os
from sklearn.metrics import *

results_directory = '/sb-personal/cvqa/results/c2vqa-verbs-results-final'

output_directory = '/sb-personal/cvqa/src/c2vqa-verbs/analysis'
output_joined_file = os.path.join(output_directory, "all_models_test_results.csv")

if os.path.exists(output_joined_file):
	all_df = pd.read_csv(output_joined_file)
else:
	all_df = None

	for f in os.listdir(results_directory):
		if not f.endswith('test_results.csv'):
			continue

		print f
		parts = f.split('-')
		model = parts[0]
		trial = parts[1]
		path = os.path.join(results_directory, f)
		df = pd.read_csv(path)

		if 'y_predict_relevance' in df:
			df['y_predict_relevance'] = df['y_predict_relevance'].apply(lambda x: 0 if x == "[False]" else 1)
		else:
			df['y_predict_relevance'] = df['y_predict'].apply(lambda x: 0 if x.startswith("no ") else 1)

		y_true = np.asarray(df['relevant'].tolist())
		y_predict = np.asarray(df['y_predict_relevance'].tolist())

		df['model'] = model
		df['trial'] = trial
		if all_df is None:
			all_df = df
		else:
			all_df = all_df.append(df)

	print all_df

	all_df.to_csv(output_joined_file)

all_df = all_df[all_df['trial'] == 0]
lstm_model_df = all_df[all_df['model'] == 'avg']
qcatt_model_df = all_df[all_df['model'] == 'qcatt']

qcatt_correct_df = qcatt_model_df[qcatt_model_df['y_predict'] == qcatt_model_df['y_true']]
lstm_incorrect_df = lstm_model_df[lstm_model_df['y_predict'] != lstm_model_df['y_true']]

print len(qcatt_correct_df)
print len(lstm_incorrect_df)

best_df = qcatt_correct_df[qcatt_correct_df['qa_id'].isin(lstm_incorrect_df['qa_id'])]

sample_df = best_df.sample(5)

print 'qcatt correct'

for _,row in sample_df.iterrows():
	print ''
	print row['question']
	print row['image_file']
	print row['caption']
	print '\ttruth: [%s]' % (row['y_true'])
	print '\tqcatt: [%s]' % (row['y_predict'])
	lstm_row = lstm_incorrect_df[lstm_incorrect_df['qa_id'] == row['qa_id']].iloc[0]
	# print '\tqclstm: [%s]' % (lstm_row['question']) 
	# print '\tqclstm: [%s]' % (lstm_row['y_true']) 
	print '\tqclstm: [%s]' % (lstm_row['y_predict']) 



qcatt_incorrect_df = qcatt_model_df[qcatt_model_df['y_predict'] != qcatt_model_df['y_true']]
lstm_correct_df = lstm_model_df[lstm_model_df['y_predict'] == lstm_model_df['y_true']]

print len(qcatt_incorrect_df)
print len(lstm_correct_df)

best_df = qcatt_incorrect_df[qcatt_incorrect_df['qa_id'].isin(lstm_correct_df['qa_id'])]

sample_df = best_df.sample(5)

print 'qcatt incorrect'

for _,row in sample_df.iterrows():
	print ''
	print row['question']
	print row['image_file']
	print row['caption']
	print '\ttruth: [%s]' % (row['y_true'])
	print '\tqcatt: [%s]' % (row['y_predict'])
	lstm_row = lstm_correct_df[lstm_correct_df['qa_id'] == row['qa_id']].iloc[0]
	# print '\tqclstm: [%s]' % (lstm_row['question']) 
	# print '\tqclstm: [%s]' % (lstm_row['y_true']) 
	print '\tqclstm: [%s]' % (lstm_row['y_predict']) 