import pandas as pd
import os
import random
from common import normalize_word, save_data
from utils import read_json
import numpy as np
import time
import h5py

output_dir = os.path.join('/sb-personal/cvqa/data/visual-genome/8-26-2017/generated-data/')
# questions_output_file = output_dir + 'question_object_data.csv'
new_questions_output_file = output_dir + 'question_action_data-v2.csv'
dataset_output_file = output_dir + 'actions_vg_expanded_dataset-v3.csv'
actions_output_file = output_dir + 'action_image_data-v2.csv'
base_dir = '/sb-personal/cvqa/'
features_dir = os.path.join(base_dir, 'data/visual-genome/8-29-2016/generated-data/vgg19_features/')
filenames_file = os.path.join(features_dir, 'filenames.json')
filter_infrequent = True

print "Reading image filenames..."
image_filenames = read_json(filenames_file)

print "Reading image features..."
image_features_h5data = h5py.File(features_dir + "vgg19_features.h5", 'r')
image_features = np.array(image_features_h5data['dataset_1'])
print '\tImage features count: [%d]' % (len(image_features))

df = pd.read_csv(new_questions_output_file)
df['relevant'] = 1
print 'Dataset size: [%d]' % (len(df))

actions_df = pd.read_csv(actions_output_file)
print 'Actions file size: [%d]' % (len(actions_df))

qa_ids = df['qa_id'].unique().tolist()

new_questions = []
total = len(qa_ids)

all_image_files = set(actions_df['image_file'].unique().tolist())

#Only pick from images that we have
# all_image_files = all_image_files - set(df['image_file'].unique().tolist())

for i,qa_id in enumerate(qa_ids):
	if i % 1000 == 0:
		print "Question: [%d/%d]" % (i,total)
		# save_data(new_questions, dataset_output_file)
	rows = df[df['qa_id'] == qa_id]
	question_actions = rows['question_action'].unique().tolist()
	question_actions = [normalize_word(o) for o in question_actions]
	question_actions.sort()
	question_actions = set(question_actions)

	# print question_actions

	images_with_actions = set(actions_df[actions_df['action'].isin(question_actions)]['image_file'].unique().tolist())

	# print images_with_actions
	images_without_actions = all_image_files - images_with_actions
	
	# #Skip this question if we can't find an irrelevant image
	if len(images_without_actions) == 0:
		print 'Skipping'
		continue
	random_image_without_actions = random.sample(images_without_actions, 1)[0] 
	
	relevant = rows.T.to_dict().values()[0]
	del relevant['question_action']
	del relevant['matched_image_action']
	del relevant['image_id']
	del relevant['Unnamed: 0']
	relevant['question_actions'] = ', '.join(question_actions)
	new_questions.append(relevant)

	# # print relevant

	irrelevant = {}
	irrelevant.update(relevant)
	irrelevant['relevant'] = 0
	irrelevant['original_answer'] = 'no ' + ' or '.join(question_actions) + ' found'
	irrelevant['answer'] = irrelevant['original_answer']
	irrelevant['image_file'] = random_image_without_actions
	irrelevant['qa_id'] = -1 * irrelevant['qa_id']
	new_questions.append(irrelevant)

	# print irrelevant

print 'Finished adding questions'
new_questions_df = save_data(new_questions, dataset_output_file)
# print new_questions_df
print len(new_questions_df)

if filter_infrequent:
	print 'Filtering dataset to remove infrequent answers...'
	irrelevant_df = new_questions_df[new_questions_df['relevant'] == 0]
	grouped_df = irrelevant_df.groupby('answer', as_index=False).count().sort_values(['image_file'])
	# print grouped_df
	# print grouped_df[grouped_df['image_file'] < 30]
	remove_answers = grouped_df[grouped_df['image_file'] < 5]['answer'].tolist()
	remove_qa_ids = new_questions_df[new_questions_df['answer'].isin(remove_answers)]['qa_id'].tolist()
	remove_qa_ids += [(-1 * qa) for qa in remove_qa_ids]
	remove_qa_ids = set(remove_qa_ids)

	new_questions_df = new_questions_df[~new_questions_df['qa_id'].isin(remove_qa_ids)]
	print 'Filtered dataset size: [%d]' % (len(new_questions_df))

new_questions_df.to_csv(dataset_output_file)