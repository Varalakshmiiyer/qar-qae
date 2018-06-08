import pandas as pd
import os
import spacy
from spacy.symbols import VERB, NOUN
import random
from pattern.en import conjugate, PROGRESSIVE, INDICATIVE
from utils import read_json
from common import save_data

print "Loading feature extractors..."
nlp = spacy.load('en')

dataset_dir = '/sb-personal/cvqa/data/visual-genome/8-29-2016/source-data/'
output_dir = os.path.join('/sb-personal/cvqa/data/visual-genome/8-26-2017/generated-data/')
dataset_output_file = output_dir + 'question_action_data-v2.csv'
editable_dataset_output_file = output_dir + 'fill_in_editable_actions_vg_expanded_dataset.csv'
output_dir = '/sb-personal/cvqa/data/visual-genome/8-26-2017/generated-data/'
output_actions_file = output_dir + 'action_image_data-v2.csv'

actions_df = pd.read_csv(output_actions_file)
# print df
all_action_names = set(actions_df['action'].tolist())

exclude = ['basketball','baseball','with','wear', 'show','look','use','dress','build','help','soccer']
exclude += ['be','remove','get','frisbee','object','clear','separate','feed','tennis','building']
exclude += ['picture','position','remote','paint',"photograph","smile"]
exclude += ['wear', 'show','use','dress','build','tennis','basketball','golf','baseball','building']
exclude_actions = set(exclude)

all_action_names = all_action_names - exclude_actions

# print all_action_names

df = pd.read_csv(dataset_output_file)

editable_questions = []

i = 0
total = len(df)
for _,row in df.iterrows():
	if i % 1000 == 0:
		print "Question: [%d/%d]" % (i,total)
	i += 1
	# print row
	image_file = row['image_file']
	image_actions = actions_df[actions_df['image_file'] == image_file]['action'].unique().tolist()
	image_actions.sort()

 	question = row['question']
 	# print question
	doc = nlp(unicode(question))
	# verbs = [(word.text, word.lemma_) for word in doc if word.pos_ == "VERB"]
	# print verbs
	question_action = row['original_question_action']

	# if question_action not in question:
	# 	print question_action
	# 	print question
	editable_question = ' '.join(["<ACTION>" if w == question_action else w for w in question.split()])

	# print ''
	# print question
	# print question_action
	# print replacement_action_conjugated
	# print editable_question

	data = {}
	data['image_file'] = image_file
	data['original_question'] = question
	data['question'] = editable_question
	data['answer'] = question_action
	# data['replacement_action'] = replacement_action_conjugated
	data['relevant'] = 0
	data['image_id'] = row['image_id']
	data['qa_id'] = -1 * row['qa_id']
	data['image_actions'] = ','.join(image_actions)
	editable_questions.append(data)

	# noedit_data = {}
	# noedit_data['image_file'] = image_file
	# noedit_data['original_question'] = question
	# noedit_data['question'] = question
	# noedit_data['answer'] = question_action
	# noedit_data['replacement_action'] = question_action
	# noedit_data['relevant'] = 1
	# noedit_data['image_id'] = row['image_id']
	# noedit_data['qa_id'] = row['qa_id']
	# noedit_data['image_actions'] = row['image_actions']
	# editable_questions.append(noedit_data)

editable_df = save_data(editable_questions, editable_dataset_output_file)
# print editable_df
