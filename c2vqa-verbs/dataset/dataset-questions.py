import os
import pandas as pd
import spacy
from spacy.symbols import VERB, NOUN
from common import save_data, normalize_word
from utils import read_json
import time
import re
from nltk.tokenize import word_tokenize

def fix_word(word, word_corrections_df):
	word = re.sub("[\",()]", "", word)
	word = word.strip()
	if word.endswith("'s"):
		word = word[:-2]
	if re.match("^.*\W$", word):
		word = word[:-1]
	if re.match("^\W.*$", word):
		word = word[1:]
	if word in word_corrections_df['word'].values:
		word = word_corrections_df[word_corrections_df['word'] == word]['correction'].values[0]
	if word in ['a','an','the']:
		word = ""
	return word

def transform_question(question, word_corrections_df):
	question = question.replace("/", " and ")
	question_words = word_tokenize(question)
	new_words = []
	for word in question_words:
		word = fix_word(word, word_corrections_df)
		if len(word) > 0:
			new_words.append(word)
	return ' '.join(new_words)


def filter_word(word):
	filter = False
	if word in ['be','have','take','do','will']:
		filter = True
	elif word in ['picture','position','remote','paint',"photograph","smile"]:
		filter = True
	elif word in ['wear', 'show','use','dress','build','tennis','basketball','golf','baseball','building']:
		filter = True
	return filter

def find_similiar_words(word):
	return []

def replace_word(word):
	aliases = []
	if word == "rid":
		return "ride"
	elif word == "skateboard":
		return "skate"
	elif word in ["face","point","watch"]:
		return "look"
	elif word in ['rid','bicycle']:
		return 'ride'
	elif word in ['cross','wander','stoll','stroll','saunter','step','wlak','round']:
		return 'walk'
	elif word in ['gather']:
		return 'meet'
	elif word in ['display']:
		return 'play'
	elif word in ['hand']:
		return 'give'
	elif word in ['grip']:
		return 'hold'
	elif word in ['clothe']:
		return 'wear'
	elif word in ['serve']:
		return 'carry'
	elif word in ['seat','sitt','sunbathe']:
		return 'sit'
	elif word in ['bat']:
		return 'hit'
	elif word in ['consume']:
		return 'eat'
	elif word in ['swing']:
		return 'play'
	elif word in ['field']:
		return 'play'
	elif word in ['pitch','fling']:
		return 'throw'
	elif word in ['frown']:
		return 'cry'
	elif word in ['sip']:
		return 'drink'
	elif word in ['secure']:
		return 'ride'
	elif word in ['abuse']:
		return 'drink'
	elif word in ['travel']:
		return 'drive'
	elif word in ['wheel']:
		return 'hold'
	elif word in ['toss']:
		return 'catch'
	elif word in ['board']:
		return 'ride'
	elif word in ['congregate']:
		return 'gather'
	# elif word in ['dry']:
	# 	return 'blow']
	elif word in ['paddle']:
		return 'row'
	elif word in ['grab']:
		return 'hold'
	elif word in ['socialize','converse','speak']:
		return 'talk'
	elif word in ['drag']:
		return 'pull'
	# elif word in ['wait']:
	# 	return 'stand']
	elif word in ['jog']:
		return 'run'
	elif word in ['hunch']:
		return 'bend'
	return word

units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

def transform_answer(answer, nlp, units):
	if answer.startswith("a "):
		answer = answer[2:]
	answer = re.sub("[/&,]"," and ",answer)
	answer = re.sub("\s+"," ",answer)
	answer = answer.replace(' and and ', ' and ')
	answer = re.sub("\s+t\s+shirt\s+"," tshirt ",answer)
	answer = re.sub("cell|cellular phone"," phone ",answer)
	answer = answer.replace('  phone  phone', ' phone ')
	answer = re.sub("puppy dog"," puppy ",answer)
	# answer = answer.replace("/", " and ")
	if (re.match("\w+( and \w+)+", answer)):
		pieces = answer.split(" and ")
		pieces.sort()
		answer = ' and '.join(pieces)
	doc = nlp(unicode(answer))
	words = []
	for word in doc:
		w = word.text
		if word.pos in [NOUN,VERB]:
			w = word.lemma_
		w = normalize_word(w)
		if re.match("^\d+$", w):
			number_word = int(w)
			if number_word >= 0 and number_word < len(units):
				w = units[number_word]
		if not w in ['a','an','the',"'s","`","-","?"]:
			w = w.strip()
			words.append(w)
	# words = [w for w in answer.split() if w not in stop_words]
	answer = ' '.join(words)
	answer = answer.strip()
	if re.match("^it[s]? (be )?\w+$", answer):
		answer = answer.split(' ')[-1]
	if answer.startswith("there be "):
		answer = answer[len("there be "):]
	return answer

def get_question_type(question):
	question_type = question.split()[0]
	question_type = question_type.split("'")[0]
	if re.match("^.*[\Ws]$",question_type):
		question_type = question_type[:-1]
	if question_type.endswith("i"):
		question_type = question_type[:-1]
	return question_type

dataset_dir = '/sb-personal/cvqa/data/visual-genome/8-29-2016/source-data/'
dataset_dir_new = '/sb-personal/cvqa/data/visual-genome/8-26-2017/source-data/'
coco_a_dir = '/sb-personal/cvqa/data/coco-a/generated-data/'
output_dir = '/sb-personal/cvqa/data/visual-genome/8-26-2017/generated-data/'
output_actions_file = output_dir + 'action_image_data-v2.csv'
actions_file = coco_a_dir + 'coco_a_actions.csv'
adverbs_file = coco_a_dir + 'coco_a_adverbs.csv'
questions_output_file = output_dir + 'question_action_data-v2.csv'
people_aliases = set(pd.read_csv(output_dir + "people_aliases.csv")['aliases'])
word_corrections_df = pd.read_csv(os.path.join(output_dir, "question_word_corrections.csv"))

print "Loading feature extractors..."
nlp = spacy.load('en')

image_data = read_json(dataset_dir + 'image_data.json')
print 'Reading image data...'
imageid_to_data = {}
url_to_imageid = {}
for i, current_image_data in enumerate(image_data):
	url = os.path.basename(current_image_data['url'])
	imageid_to_data[current_image_data['image_id']] = {'url':url,'qId':i}
	url_to_imageid[url] = current_image_data['image_id']

del image_data

print 'Reading actions...'
if os.path.exists(output_actions_file):
	actions_df = pd.read_csv(output_actions_file)
else:
	actions_df = pd.read_csv(actions_file)

print '\tActions: [%d]' % (len(actions_df))

if not os.path.exists(output_actions_file):
	
	all_action_names = set(actions_df['action'].tolist())

	add_region_actions = []
	print 'Reading regions...'
	region_descriptions = read_json(dataset_dir + 'region_descriptions.json')
	total = len(region_descriptions)
	for ri, region_area in enumerate(region_descriptions):
		if ri % 1000 == 0:
			print "Region: [%d/%d]" % (ri,total)

		for region in region_area['regions']:
			doc = nlp(unicode(region['phrase']))
			people = [word.lemma_ for word in doc if word.pos_ == "NOUN" and word.lemma_ in people_aliases]
			verbs = [replace_word(word.lemma_) for word in doc if word.pos_ == "VERB"]

			if len(people) == 0 or len(verbs) == 0:
				continue
		
			image_id = region['image_id']
			url = imageid_to_data[image_id]['url']
			image_actions = set(actions_df[actions_df['image_id'] == image_id]['action'].tolist())

			actions = []
			expanded_verbs = [[verb] + find_similiar_words(verb) for verb in verbs]
			for verb_list in expanded_verbs:
				if filter_word(verb_list[0]):
					continue
				for verb in verb_list:
					if verb in all_action_names and verb not in image_actions:
						actions.append(verb)
						break

			if len(actions) == 0:
				# print ''
				# print region
				# print people
				# print verbs
				# print actions
				continue

			for action in actions:
				data = {}
				data['action'] = action
				data['image_file'] = url
				data['image_id'] = image_id
				data['coco_id'] = ""
				data['source'] = 'region'
				add_region_actions.append(data)

			# people_aliases

	del region_descriptions

	print 'Appending new actions...'

	actions_df['source'] = 'coco_a'
	actions_df = actions_df.append(pd.DataFrame(add_region_actions))

	print '\tActions: [%d]' % (len(actions_df))

all_action_names = set(actions_df['action'].tolist())
action_image_ids = set(actions_df['image_id'].tolist())

print 'Reading questions...'
questions = read_json(dataset_dir + 'question_answers.json')

add_actions = []
question_actions_data = []
total = len(questions)

found_image_ids = []

for qi, question_image_group in enumerate(questions):
	if qi % 1000 == 0:
		print "Question: [%d/%d]" % (qi,total)
	# if qi % 20000 == 0:
	# 	print 'Saving...'
	# 	save_data(question_object_data, questions_output_file)
	
	if len(question_image_group['qas']) == 0:
		continue

	image_id = question_image_group['qas'][0]['image_id']
	url = imageid_to_data[image_id]['url']
	image_actions = actions_df[actions_df['image_id'] == image_id]['action'].tolist()
	image_actions.sort()

	if image_id not in action_image_ids or len(image_actions) == 0:
		continue

	coco_id = actions_df[actions_df['image_id'] == image_id]['coco_id'].tolist()[0]

	for q in question_image_group['qas']:	
		question = q['question']
		answer = q['answer']
		question = question.lower()[:-1]
		original_question = question
		question = transform_question(question, word_corrections_df)
		question_type = get_question_type(question)
		answer = answer.lower()
		if answer[-1] == ".":
			answer = answer[:-1]
		original_answer = answer
		answer = transform_answer(answer, nlp, units)
		doc = nlp(unicode(question))
		people = [word.lemma_ for word in doc if word.pos_ == "NOUN" and word.lemma_ in people_aliases]
		verb_lemma_to_full_verb = {replace_word(word.lemma_):word.text for word in doc if word.pos_ == "VERB" and not filter_word(replace_word(word.lemma_))}
		verbs = verb_lemma_to_full_verb.keys()
		if len(people) == 0 or len(verbs) == 0:
			continue
		expanded_verbs = [[verb] + find_similiar_words(verb) for verb in verbs]
		question_actions = []
		for verb_list in expanded_verbs:
			original_text = verb_lemma_to_full_verb[verb_list[0]]
			for verb in verb_list:
				if verb in image_actions:
					verb_lemma_to_full_verb[verb] = original_text
					question_actions.append((verb_list[0],verb))
					break
				elif verb in all_action_names:
					# print '\tMissing action found: [%s]' % (verb)
					verb_lemma_to_full_verb[verb] = original_text
					data = {}
					data['action'] = verb
					data['image_file'] = url
					data['image_id'] = image_id
					data['coco_id'] = coco_id
					data['source'] = 'question'
					add_actions.append(data)
					question_actions.append((verb,verb))
					break

		if len(question_actions) == 0:
			# print ''
			# print question
			# print expanded_verbs
			# print image_actions
			break

		if image_id not in found_image_ids:
			found_image_ids.append(image_id)

		for o in question_actions:
			data = {}
			data['original_question'] = original_question
			data['question'] = question
			data['original_answer'] = original_answer
			data['answer'] = answer
			data['image_file'] = url
			data['qa_id'] = q['qa_id']
			data['question_type'] = question_type
			data['image_id'] = image_id
			data['question_action'] = o[0]
			data['matched_image_action'] = o[1]
			data['original_question_action'] = verb_lemma_to_full_verb[o[1]]
			data['image_actions'] = ','.join(image_actions)
			question_actions_data.append(data)

print "Finished questions"

# print add_actions

question_actions_df = save_data(question_actions_data, questions_output_file)
print "Question actions: [%d]" % (len(question_actions_df))

if not os.path.exists(output_actions_file):
	print 'Adding in new actions...'
	add_df = pd.DataFrame(add_actions)
	add_df = add_df.loc[:, ~add_df.columns.str.contains('^Unnamed')]
	# print add_df
	# print actions_df
	actions_df = actions_df.append(add_df)
actions_df = actions_df.loc[:, ~actions_df.columns.str.contains('^Unnamed')]
actions_df = actions_df[actions_df['image_id'].isin(found_image_ids)]
actions_df.to_csv(output_actions_file)