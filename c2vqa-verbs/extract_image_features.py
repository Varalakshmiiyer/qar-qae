from keras.preprocessing import image
from keras.models import Model
import numpy as np
import os
import h5py
import psutil
import json

from common import save_features

def print_memory():
	print "Memory: [%f]"  % (psutil.virtual_memory()[2])

def write_state(state, output_path):
	with open(output_path, "w") as f:
		f.write(str(state))

# https://gogul09.github.io/software/flower-recognition-deep-learning

type_name = "resnet50"

images_path = '/sb-personal/cvqa/data/visual-genome/8-29-2016/source-data/images'
output_dir = '/sb-personal/cvqa/data/visual-genome/8-29-2016/generated-data/%s_features' % (type_name)
features_output_path = output_dir + "/%s_features.h5" % (type_name)
filenames_output_path = output_dir + "/filenames.json"
state_output_path = output_dir + "/state.txt"

if type_name == "vgg19":
	from keras.applications.vgg19 import VGG19
	from keras.applications.vgg19 import preprocess_input
	base_model = VGG19(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
elif type_name == "resnet50":
	from keras.applications.resnet50 import ResNet50
	from keras.applications.resnet50 import preprocess_input
	model = ResNet50(weights='imagenet', include_top = False)
	# model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

print "Reading image filenames..."
filenames = [f for f in os.listdir(images_path) if f.endswith(".jpg")]

with open(filenames_output_path, "w") as open_file:
	json.dump({f:i for i,f in enumerate(filenames)}, open_file)

state = 0

if not os.path.exists(state_output_path):
	write_state(state, state_output_path)

with open(state_output_path) as f:
	state = int(f.read())

print state

features = []
files = []

if os.path.exists(features_output_path):
	print 'Reading %s...' % (features_output_path)
	image_features_h5data = h5py.File(features_output_path, 'r')
	features = np.array(image_features_h5data['dataset_1']).tolist()
	image_features_h5data.close()

total = len(filenames) - 1
for index, image_file in enumerate(filenames[state:]):
	index += state
	print "[%d/%d]" % (index, total)
	img_path = os.path.join(images_path, image_file)

	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	feature = model.predict(x).flatten()
	# print feature.shape
	features.append(feature)
	if index % 50 == 0:
		print "\tSaving features..."
		print_memory()
		save_features(features, features_output_path)
		write_state(index, state_output_path)

print "\tSaving features..."
print_memory()
save_features(features, features_output_path)