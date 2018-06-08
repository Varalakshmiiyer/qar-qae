import pandas as pd
from utils import read_json
import h5py
import numpy as np

def save_features(features, output_path):
	h5f_data = h5py.File(output_path, 'w')
	h5f_data.create_dataset('dataset_1', data=np.array(features))
	h5f_data.close()

def save_data(data, output_file):
	# print 'Data items: [%d]' % (len(data))
	df = pd.DataFrame(data)
	df.to_csv(output_file)
	return df

def filter_word(word):
	filter = False
	if word in ["color","type","who","what","when","where","why","how","day","background","foreground","parking","stripe","weather","scene","time"]:
		filter = True
	if word in ["photo","photograph","picture","painting"]:
		filter = True
	return filter

def normalize_word(w):
	if w == "upon":
		w = "on"
	if w == "teelth":
		w = "teelth"
	if w == "n't":
		w = "not"
	if w == "ziplin":
		w = "zipline"
	if w == "ax":
		w = "axe"
	if w == "brasil":
		w = "brazil"
	if w in ['buidling','buiding']:
		w = "building"
	if w == "carrige":
		w = "carriage"
	if w == "chevy":
		w = "chevrolet"
	if w in ['sedan','car','coupe']:
		w = "car"
	if w in ["street","road"]:
		w = "road"
	if w in ["shirt","tshirt","teeshirt","t-shirt"]:
		w = "shirt"
	if w in ['motorbike','motorcycle']:
		w = "motorcycle"
	if w in ['dude','guy','man','male']:
		w = "man"
	if w in ['gal','lady','woman','female']:
		w = "woman"
	if w in ['underpant','underwear']:
		w = "underwear"
	if w in ["sandwhich","sandwich"]:
		w = "sandwich"
	if w in ["mountainside","mountain"]:
		w = "mountain"
	if w in ["kitty","kitten"]:
		w = "kitten"
	if w in ["doughnut","donut"]:
		w = "doughnut"
	if w in ["sofa","couch"]:
		w = "couch"
	if w in ["bicycyle","bicycle"]:
		w = "bicycle"
	if w in ["kid","child"]:
		w = "child"
	return w

def get_people_aliases():
	output_dir = '/sb-personal/cvqa/data/visual-genome/8-26-2017/generated-data/'
	people_aliases_df = pd.read_csv(output_dir + 'people_aliases.csv')
	return people_aliases_df['aliases'].unique().tolist()

people_aliases = get_people_aliases()

def find_similiar_words(word):
	aliases = []
	if word in people_aliases:
		aliases += ["man","woman","child","girl","boy","guy","dude","lady","gal"] + people_aliases
	elif word in ["adult"]:
		aliases += ["man","woman"] + people_aliases
	elif word in ["kid","child","baby"]:
		aliases += ["child","girl","boy","kid","male_child","baby"]
	elif word in ["sedan","license"]:
		aliases += ["car"]
	elif word in ["guy","dude","boy","man"]:
		aliases += ["man","boy","male_child"] + people_aliases
	elif word in ["lady","gal","girl","woman"]:
		aliases += ["woman", "girl"] + people_aliases
	elif word in ["grandma","grandmother"]:
		aliases += ["woman"]
	elif word in ["grandpa","grandfather"]:
		aliases += ["man"]
	elif word in ["vehicle"]:
		aliases += ["car","truck","van"]
	elif word in ["bike"]:
		aliases += ["bicycle","motorcycle"]
	elif word in ["boot","sneaker","heel","shoe"]:
		aliases += ["shoe","gym_shoe"]
	elif word in ["pant","trouser","jean"]:
		aliases += ["trouser","jean","pant"]
	elif word in ["top","shirt"]:
		aliases += ["top","shirt","tank_top"]
	elif word in ["jacket","coat"]:
		aliases += ["jacket","coat"]
	elif word in ["monitor","computer_monitor","screen"]:
		aliases += ["monitor","computer_monitor","screen","computer_screen"]
	elif word in ["clothe","clothes","clothing"]:
		aliases += ["apparel","shirt","jacket","pants","coat"]
	elif word in ["street","road"]:
		aliases += ["street","road"]
	elif word in ["shop","store","mercantile_establishment"]:
		aliases += ["shop","store","mercantile_establishment"]
	elif word in ["carpet","rug","floor","tile","ground"]:
		aliases += ["carpet","rug","floor","tile","ground"]
	elif word in ["tv"]:
		aliases += ["television"]
	elif word in ["dresser","cabinet"]:
		aliases += ["dresser","cabinet"]
	elif word in ["furniture"]:
		aliases += ["dresser","cabinet","sofa","couch","chair","recliner","bed"]
	elif word in ["bookshelf","shelf"]:
		aliases += ["bookshelf","shelf"]
	elif word in ["blind"]:
		aliases += ["window_blind"]
	elif word in ["chair","seat","couch","sofa","armchair"]:
		aliases += ["chair","seat","couch","sofa","armchair"]
	elif word in ["table","desk","desktop","tablecloth"]:
		aliases += ["table","desk","desktop","tablecloth"]
	elif word in ["lamp","lampshade","lampstand"]:
		aliases += ["lamp","lampshade","lampstand"]
	elif word in ["bear","teddy"]:
		aliases += ["teddy","teddy_bear"]
	elif word in ["cpus","cpu","pc","pcs","computer"]:
		aliases += ["central_processing_unit","cpus","cpu","pc","pcs","computer"]
	elif word in ["mug","cup","bottle"]:
		aliases += ["mug","cup","glass","bottle","water_bottle","coffee_mug"]
	elif word in ["spectacle","glass","sunglass"]:
		aliases += ["spectacle","glass","sunglass"]
	elif word in ["earphone","headphone"]:
		aliases += ["earphone","headphone"]
	elif word in ["puzzle","puzze","crossword"]:
		aliases += ["crossword_puzzle"]
	elif word in ["gazette"]:
		aliases += ["newspaper","news"]
	elif word in ["cloth","fabric"]:
		aliases += ["cloth","fabric"]
	elif word in ["phone","telephone"]:
		aliases += ["phone","telephone"]
	elif word in ["pathway","sidewalk"]:
		aliases += ["pathway","sidewalk"]
	elif word in ["bush","tree","plant","vine","flower","branch"]:
		aliases += ["bush","tree","plant","vine","flower","branch"]
	elif word in ["railing"]:
		aliases += ["rail"]
	elif word in ["placard"]:
		aliases += ["sign"]
	elif word in ["luggage"]:
		aliases += ["bag"]
	return aliases

def create_object_data(objects_input_file, objects_output_file, imageid_to_data, nlp):
	object_image_data = []
	objects = read_json(objects_input_file)

	total = len(objects)

	print 'Total objects: [%d]' % (total)
	print 'Reading object data...'

	for i,row in enumerate(objects):
		if i % 1000 == 0:
			print 'Objects: [%d/%d]' % (i, total)
		# stat('Objects: [%d/%d]' % (i,total))
		image_id = row['image_id']
		image_file = imageid_to_data[image_id]['url']
		image_objects = []
		for o in row['objects']:
			if len(o['synsets']) > 0:
				image_objects += [s.split(".")[0].lower() for s in o['synsets']]
		image_objects = list(set(image_objects))
		image_objects_aliases = []
		for image_object in image_objects:
			doc = nlp(unicode(image_object))
			image_object = ' '.join([word.lemma_ for word in doc])
			image_objects_aliases += find_similiar_words(image_object)
			object_image_data.append({'image_id':image_id,'image_file':image_file,'object':image_object})

		for image_object_alias in image_objects_aliases:
			object_image_data.append({'image_id':image_id,'image_file':image_file,'object':image_object_alias})

	print 'Finished object data'

	del objects

	objects_df = pd.DataFrame(object_image_data)
	objects_df.to_csv(objects_output_file)
	return objects_df