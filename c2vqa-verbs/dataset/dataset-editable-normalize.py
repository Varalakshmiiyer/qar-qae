import os
import pandas as pd

output_dir = os.path.join('/sb-personal/cvqa/', 'data/visual-genome/8-26-2017/generated-data/')
questions_output_file = os.path.join(output_dir, 'editable_and_not_editable_actions_vg_expanded_dataset-v3.csv')
new_questions_output_file = os.path.join(output_dir, 'normalized_editable_and_not_editable_actions_vg_expanded_dataset-v3.csv')

new_questions_df = pd.read_csv(questions_output_file)

grouped_df = new_questions_df.groupby('answer', as_index=False).count().sort_values(['image_file'])
print grouped_df[['answer','image_file']]

# print len(new_questions_df)

reduce_list = ['no edit because holding','edit to holding','edit to standing','no edit because standing','no edit because sitting','edit to sitting']
for reduce_item in reduce_list:
	length = len(new_questions_df[new_questions_df['answer'] == reduce_item])
	remove_qa_ids = new_questions_df[new_questions_df['answer'] == reduce_item].sample(length-4000)['qa_id'].tolist()
	remove_qa_ids += [(-1 * qa) for qa in remove_qa_ids]
	remove_qa_ids = set(remove_qa_ids)
	new_questions_df = new_questions_df[~new_questions_df['qa_id'].isin(remove_qa_ids)]

# print len(new_questions_df)

# grouped_df = new_questions_df.groupby('answer', as_index=False).count().sort_values(['image_file'])
# print grouped_df[['answer','image_file']]

grouped_df = new_questions_df.groupby('answer', as_index=False).count().sort_values(['image_file'])
remove_answers = grouped_df[grouped_df['image_file'] < 3]['answer'].tolist()
remove_qa_ids = new_questions_df[new_questions_df['answer'].isin(remove_answers)]['qa_id'].tolist()
remove_qa_ids += [(-1 * qa) for qa in remove_qa_ids]
remove_qa_ids = set(remove_qa_ids)
new_questions_df = new_questions_df[~new_questions_df['qa_id'].isin(remove_qa_ids)]

grouped_df = new_questions_df.groupby(['question','answer'], as_index=False).count().sort_values(['image_file'])
print grouped_df[['question','answer','image_file']]

# print len(new_questions_df)
new_questions_df = new_questions_df.copy()
new_questions_df.to_csv(new_questions_output_file)