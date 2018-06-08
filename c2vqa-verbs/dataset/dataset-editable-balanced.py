import pandas as pd
import os
import spacy
from spacy.symbols import VERB, NOUN
import random
from pattern.en import conjugate, PROGRESSIVE, INDICATIVE
from utils import read_json
from common import save_data

def create_template_question(row):
	question = row['question']
	question_action = row['original_question_action']
	return ' '.join(["<ACTION>" if w == question_action else w for w in question.split()])


dataset_dir = '/sb-personal/cvqa/data/visual-genome/8-29-2016/source-data/'
output_dir = os.path.join('/sb-personal/cvqa/data/visual-genome/8-26-2017/generated-data/')
dataset_output_file = output_dir + 'question_action_data-v2.csv'
editable_dataset_output_file = output_dir + 'balanced_editable_actions_vg_expanded_dataset.csv'
output_dir = '/sb-personal/cvqa/data/visual-genome/8-26-2017/generated-data/'
output_actions_file = output_dir + 'action_image_data-v2.csv'

actions_df = pd.read_csv(output_actions_file)
# print df

exclude = ['pointing']
exclude_actions = set(exclude)


df = pd.read_csv(dataset_output_file)
df['question_template'] = df.apply(lambda row: create_template_question(row), axis=1)

print df

all_image_files = set(actions_df['image_file'].unique().tolist())

editable_questions = []

i = 0
total = len(df)
for _,row in df.iterrows():
	if i % 1000 == 0:
		print "Question: [%d/%d]" % (i,total)
	i += 1
	question_action = row['original_question_action']

	if question_action in exclude_actions:
		continue

	images_with_actions = set(actions_df[actions_df['action'].isin(question_action.split(','))]['image_file'].unique().tolist())

	images_with_question_template = df[df['question_template'] == row['question_template']]
	images_with_question_template = images_with_question_template[images_with_question_template['original_question_action'] != row['original_question_action']]
	images_with_question_template = images_with_question_template[~images_with_question_template['original_question_action'].isin(exclude_actions)]
	images_with_question_template = set(images_with_question_template['image_file'].unique().tolist())
	images_without_actions = images_with_question_template - images_with_actions

	# #Skip this question if we can't find an irrelevant image
	if len(images_without_actions) == 0:
		# print 'Skipping'
		continue
	
 	question = row['question']

	random_image_without_actions = random.sample(images_without_actions, 1)[0] 

	irrelevant_row = df[df['question_template'] == row['question_template']]
	irrelevant_row = irrelevant_row[irrelevant_row['image_file'] == random_image_without_actions].iloc[0]

	if irrelevant_row['original_question_action'] in exclude_actions:
		continue
	

	data = {}
	data['image_file'] = random_image_without_actions
	data['question'] = question
	data['answer'] = 'edit to ' + irrelevant_row['original_question_action']
	data['relevant'] = 0
	data['qa_id'] = -1 * row['qa_id']
	
	image_actions = actions_df[actions_df['image_file'] == random_image_without_actions]['action'].unique().tolist()
	image_actions.sort()
	data['image_actions'] = ','.join(image_actions)
	editable_questions.append(data)

	image_file = row['image_file']

	noedit_data = {}
	noedit_data['image_file'] = image_file
	noedit_data['question'] = question
	noedit_data['answer'] = 'no edit because ' + question_action
	noedit_data['relevant'] = 1
	noedit_data['qa_id'] = row['qa_id']
	image_actions = actions_df[actions_df['image_file'] == image_file]['action'].unique().tolist()
	image_actions.sort()
	noedit_data['image_actions'] = ','.join(image_actions)
	editable_questions.append(noedit_data)

editable_df = save_data(editable_questions, editable_dataset_output_file)
print editable_df
