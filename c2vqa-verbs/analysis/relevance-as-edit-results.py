import pandas as pd

def find_match(row):
	return (row['y_true'] in row['y_predict_5'].split(','))

results_dir = '/sb-personal/cvqa/results/c2vqa-verbs-results-relevance_as_edit/'
stats_df = pd.read_csv(results_dir + 'stats.csv')

print stats_df.groupby(['model_name']).mean()

qcatt = pd.read_csv(results_dir + 'qcatt-0-test_results.csv')
qc = pd.read_csv(results_dir + 'qc-0-test_results.csv')
qclstm = pd.read_csv(results_dir + 'qclstm-0-test_results.csv')

qcatt.columns.values[0] = 'question_number'
qc.columns.values[0] = 'question_number'
qclstm.columns.values[0] = 'question_number'

qcatt['matches'] = qcatt.apply(lambda row: find_match(row), axis=1)
qc['matches'] = qc.apply(lambda row: find_match(row), axis=1)
qclstm['matches'] = qclstm.apply(lambda row: find_match(row), axis=1)

qcatt_matched_df = qcatt[qcatt['matches'] == False]
qc_matched_df = qc[qc['matches'] == True]
qclstm_matched_df = qclstm[qclstm['matches'] == True]

exclude_numbers = set(qclstm_matched_df['question_number'].unique().tolist() + qc_matched_df['question_number'].unique().tolist())
matched_df = qcatt_matched_df[qcatt_matched_df['question_number'].isin(exclude_numbers)]

for _, row in matched_df.sample(10).iterrows():
	bad_qc = qc[qc['question_number'] == row['question_number']]
	bad_qclstm = qclstm[qclstm['question_number'] == row['question_number']]
	if len(bad_qc) == 0 or len(bad_qclstm) == 0:
		continue

	print ''
	print row['original_question']
	print row['image_file']
	print 'True: [%s]' % (row['y_true'])
	print 'qcatt'
	print '\t' + row['y_predict_5']
	print 'qc'
	print '\t' + bad_qc.iloc[0]['y_predict_5']
	print 'qclstm'
	print '\t' + bad_qclstm.iloc[0]['y_predict_5']