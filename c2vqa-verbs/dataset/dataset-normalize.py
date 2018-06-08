import os
import pandas as pd

output_dir = os.path.join('/sb-personal/cvqa/', 'data/visual-genome/8-26-2017/generated-data/')
questions_output_file = os.path.join(output_dir, 'actions_vg_expanded_dataset-v3.csv')
new_questions_output_file = os.path.join(output_dir, 'specific_relevance_actions_vg_expanded_dataset-v2.csv')

df = pd.read_csv(questions_output_file)

print len(df)

reduce_list = ['no hold found','no stand found','no sit found']
for reduce_item in reduce_list:
	length = len(df[df['answer'] == reduce_item])
	remove_qa_ids = df[df['answer'] == reduce_item].sample(length-2000)['qa_id'].tolist()
	remove_qa_ids += [(-1 * qa) for qa in remove_qa_ids]
	remove_qa_ids = set(remove_qa_ids)
	df = df[~df['qa_id'].isin(remove_qa_ids)]

grouped_df = df.groupby('answer', as_index=False).count().sort_values(['image_file'])
print grouped_df[['answer','image_file']]

print len(df)

df = df.copy()

df['specific_answer'] = ''
i = 0
total = len(df)
for _,row in df[df['qa_id'] < 0].iterrows():
	if i == 1000:
		print 'Question: [%d/%d]' % (i,total)
	qa_id = -1 * row['qa_id']
	# print qa_id
	specific_answer = row['answer'][3:-7]
	df.loc[df['qa_id'] == qa_id, 'answer'] = 'relevant because ' + row['answer'][3:]
	df.loc[df['qa_id'] == qa_id, 'specific_answer'] = specific_answer

	row['specific_answer'] = specific_answer
	i+=1
	# print df[df['qa_id'] == qa_id]
	# df[df['qa_id']==qa_id]['answer'] = answer 

# print df
df.to_csv(new_questions_output_file)
# grouped_df = df.groupby('answer', as_index=False).count().sort_values(['image_file'])
# print grouped_df[['answer','image_file']]

# df.to_csv(os.path.join(output_dir, 'sub_relevance_actions_vg_expanded_dataset.csv'))