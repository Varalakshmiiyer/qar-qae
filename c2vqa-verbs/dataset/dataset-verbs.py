import pandas as pd 
import os
import sys
from utils import read_json

def intersect(a,b):
	return list(set(a) & set(b))

vg_dataset_dir = '/sb-personal/cvqa/data/visual-genome/8-29-2016/source-data/'
vqa_data_path = '/sb-personal/cvqa/data/vqa/1.0/'
coco_a_path = '/sb-personal/cvqa/data/coco-a/'
verse_data_path = '/sb-personal/cvqa/data/verse-dataset/'
verbs = read_json(coco_a_path + 'source-data/cocoa_beta2015.json')
verb_list = read_json(coco_a_path + 'source-data/visual_verbnet_beta2015.json')
verse_annotations_df = pd.read_csv(verse_data_path + 'gold_all_final_lemma_image_sense_annotations.csv')

output_actions_file = coco_a_path + "generated-data/coco_a_actions.csv"
output_adverbs_file = coco_a_path + "generated-data/coco_a_adverbs.csv"

image_data = read_json(vg_dataset_dir + 'image_data.json')
print 'Reading VG image data...'
imageid_to_data = {}
url_to_imageid = {}
for i, current_image_data in enumerate(image_data):
	url = os.path.basename(current_image_data['url'])
	cocoid = current_image_data['coco_id']
	if cocoid == None:
		continue
	imageid_to_data[str(cocoid).zfill(12)] = {'url':url,'qId':i, 'image_id':current_image_data['image_id']}
	url_to_imageid[url] = current_image_data['image_id']

del image_data

print 'Verse images: [%d]' % (len(imageid_to_data))

# vqa = read_json(vqa_data_path + 'source-data/OpenEnded_mscoco_train2014_questions.json')

# outputVerbs = vqaDataFolder + 'generated-data/verbs-vqa.json'
# outputAllVerbs = vqaDataFolder + 'generated-data/all-verbs-vqa.csv'

verse_annotations_df = verse_annotations_df[verse_annotations_df['COCO/TUHOI']=='COCO']
verse_cocoids = [a.split('_')[-1][:-4] for a in verse_annotations_df['image'].unique().tolist()]
print 'Verse images: [%d]' % (len(verse_cocoids))

image_data = ['captions_val2014.json','captions_train2014.json']

vqa_image_data = {}
for i in image_data:

	data = read_json(vqa_data_path + 'source-data/' + i)
	images = data['images']
	for image in images:
		vqa_image_data[str(image['id']).zfill(12)] = image

print 'VQA image ids: [%d]' % (len(vqa_image_data))

actionid_to_action = {}
actionid_to_adverb = {}
for a in verb_list['visual_adverbs']:
	actionid_to_adverb[a['id']] = a['name']
for a in verb_list['visual_actions']:
	actionid_to_action[a['id']] = a['name']

cocoids_to_actions = {}
cocoids_to_adverbs = {}
for index in verbs['annotations']:
	for annotation in verbs['annotations'][index]:
		# print annotation
		cocoid = str(annotation['image_id']).zfill(12)
		if cocoid not in cocoids_to_actions:
			cocoids_to_actions[cocoid] = []
		if cocoid not in cocoids_to_adverbs:
			cocoids_to_adverbs[cocoid] = []
		cocoids_to_actions[cocoid] += [s for a in annotation['visual_actions'] for s in actionid_to_action[a].split("_")]
		cocoids_to_adverbs[cocoid] += [actionid_to_adverb[a] for a in annotation['visual_adverbs']]

action_data = []
for c in cocoids_to_actions:
 	items = list(set(cocoids_to_actions[c]))
 	items.sort()
	if c not in imageid_to_data:
		continue
	image_data = imageid_to_data[c]
	for i in items:
		data = {}
		data['action'] = i
		data['image_file'] = image_data['url']
		data['image_id'] = image_data['image_id']
		data['coco_id'] = c
		action_data.append(data)

action_df = pd.DataFrame(action_data)
# print action_df
action_df.to_csv(output_actions_file)

adverb_data = []
for c in cocoids_to_adverbs:
 	items = list(set(cocoids_to_adverbs[c]))
 	items.sort()
	if c not in imageid_to_data:
		continue
	image_data = imageid_to_data[c]
	for i in items:
		data = {}
		data['adverb'] = i
		data['image_file'] = image_data['url']
		data['coco_id'] = c
		adverb_data.append(data)

adverb_df = pd.DataFrame(adverb_data)
# print adverb_df
adverb_df.to_csv(output_adverbs_file)

print 'COCO image ids: [%d]' % (len(cocoids_to_actions))

