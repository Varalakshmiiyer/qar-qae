import pandas as pd
import os
import re
from vocab import Vocabulary
import numpy as np
import h5py
import spacy
from models import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
from utils import read_json
from features import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import *
from random import shuffle

def save_data(data, output_file):
	# print 'Data items: [%d]' % (len(data))
	df = pd.DataFrame(data)
	df.to_csv(output_file)
	return df

import itertools
import matplotlib.pyplot as plt
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_predict

#https://github.com/anantzoid/VQA-Keras-Visual-Question-Answering

def load_word_vectors(word_vectors_file, vocab):
    f = open(word_vectors_file,'r')
    model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        if (word not in vocab.word_index):
        	continue
        embedding = [float(val) for val in split_line[1:]]
        model[word] = embedding
    print "Finished loading [%d] word vectors" % len(model)
    return model

def load_word_data(questions_df, image_captions, exclude_word_list):
	vocab = Vocabulary()
	answers = Vocabulary(first_word="RELEVANT")
	specific_answers = Vocabulary()
	question_seq_length = 1
	caption_seq_length = 1
	
	print "Generating vocabulary and answer indices..."
	new_questions = []
	for _, row in questions_df.iterrows():
		question_words = row['question'].split(' ')
		
		if len(question_words) > question_seq_length:
			question_seq_length = len(question_words)

		all_words = question_words

		image_file = row['image_file']
		if image_file in image_captions:
			caption = image_captions[image_file]
			caption_words = caption.split(' ')
			if len(caption_words) > caption_seq_length:
				caption_seq_length = len(caption_words)	
			all_words += caption_words

		for word in all_words:
			if len(word) > 0 and word not in exclude_word_list:
				vocab.add_word(word)
		# if row['relevant'] == 0:
		answers.add_word(row['answer'])
		specific_answers.add_word(row['specific_answer'])

	print '\tVocab count: [%d]' % (len(vocab))
	print '\tAnswers count: [%d]' % (len(answers))
	print '\tQuestion sequence length: [%d]' % (question_seq_length)
	print '\tCaption sequence length: [%d]' % (caption_seq_length)

	print "Loading word vectors..."
	word_to_vector = load_word_vectors(word_vectors_file, vocab)

	print 'Creating embedding matrix...'
	embedding_matrix = np.zeros((len(vocab), embedding_dim))

	words_not_found = []
	for word, i in vocab.word_index.items():
		if word not in word_to_vector:
			words_not_found.append(word)
			continue
		embedding_vector = word_to_vector[word]
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	if len(words_not_found) > 0:
		print "Words not found:","\n\t",words_not_found
		for word in words_not_found:
			del vocab.index_word[vocab.word_index[word]]

	return vocab, answers, specific_answers, embedding_matrix, word_to_vector, question_seq_length, caption_seq_length

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.plasma):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
    	pass
        # print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=4, rotation=90)
    plt.yticks(tick_marks, classes, fontsize=4)

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', type=str, default='/sb-personal/cvqa/')
	parser.add_argument('--epoch', type=int, default=30)
	parser.add_argument('--filter_images', type=int, default=-1)
	parser.add_argument('--trials', type=int, default=40)
	parser.add_argument('--model_names', type=str, default='avg,qcatt,image,avg,bow,qonly,conly')
	args = parser.parse_args()
	base_dir = args.base_dir
	image_feature_data = {'resnet50':{'size':2048}, 'vgg19':{'size':4096}}
	word_vectors_file = os.path.join(base_dir, 'data/glove/glove.840B.300d.txt')
	output_dir = os.path.join(base_dir, 'data/visual-genome/8-26-2017/generated-data/')
	images_path = os.path.join(base_dir, 'data/visual-genome/8-29-2016/source-data/images')
	caption_file = os.path.join(base_dir, 'data/cvqa/imagecaptions-vg.json')
	action_image_file = os.path.join(output_dir,'action_image_data-v2.csv')
	exclude_word_list = ['is','a','the','what','that','to','who','why']
	# questions_output_file = output_dir + 'question_object_data.csv'
	# questions_output_file = os.path.join(output_dir, 'question_object_data_filtered_lessthan_30_question_type_what_where_who.csv')
	# questions_output_file = os.path.join(output_dir, 'actions_vg_expanded_dataset.csv')
	questions_output_file = os.path.join(output_dir, 'specific_relevance_actions_vg_expanded_dataset-v2.csv')
	results_dir = os.path.join(base_dir, "results/c2vqa-verbs-results/")
	output_stats_file = os.path.join(results_dir, "stats.csv")

	# model_names = ['image','qc','bow', 'qonly','conly']
	model_names = args.model_names.split(',')

	print 'Using models: %s' % (model_names)

	embedding_dim = 300
	dropout_rate = 0.5
	number_of_epochs = args.epoch
	batch_size = 100

	print 'Loading list of all actions...'
	actions_df = pd.read_csv(action_image_file)
	all_action_names = actions_df['action'].unique().tolist()
	all_action_names.sort()
	print "\tNumber of actions: [%d]" % (len(all_action_names))
	
	print "Loading Captions generated by a Pre-Trained Captioning Model for Images..."
	image_captions = read_json(caption_file)

	print "Loading questions..."
	questions_df = pd.read_csv(questions_output_file)
	print "\tNumber of questions: [%d]" % (len(questions_df))

	vocab, answers, specific_answers, embedding_matrix, word_to_vector, question_seq_length, caption_seq_length = load_word_data(questions_df, image_captions, exclude_word_list)

	max_seq_length = max(question_seq_length,caption_seq_length)

	print "Reading image features..."
	for image_feature_name in image_feature_data:
		features_dir = os.path.join(base_dir, 'data/visual-genome/8-29-2016/generated-data/%s_features/' % (image_feature_name))
		image_features_h5data = h5py.File(features_dir + "%s_features.h5" % (image_feature_name), 'r')
		image_feature_data[image_feature_name]['data'] = np.array(image_features_h5data['dataset_1'])
		print '\tImage features count: [%d]' % (len(image_feature_data[image_feature_name]['data']))
		filenames_file = os.path.join(features_dir, 'filenames.json')

	print "Reading image filenames..."
	image_filenames = read_json(filenames_file)

	all_image_files = questions_df['image_file'].unique().tolist()
	print '\tImage files count (before caption filtering): [%d]' % (len(all_image_files))
	print "Removing images without captions..."
	all_image_files = [i for i in all_image_files if i in image_captions]

	if (args.filter_images > 0):
		all_image_files = all_image_files[:args.filter_images]
	print '\tImage files count: [%d]' % (len(all_image_files))

	all_stats = []

	trials = args.trials

	def create_model_and_data(model_name, image_files):
		if model_name == 'image':
			image_data = image_feature_data['resnet50']
			x, y, df = extract_image_features(image_data['size'], image_files, image_filenames, image_data['data'], questions_df, word_to_vector, answers, vocab, question_seq_length)
			model = vqa_model(image_data['size'], embedding_matrix, len(vocab), embedding_dim, question_seq_length, dropout_rate, len(answers))
		elif model_name == 'imageact':
			image_data = image_feature_data['resnet50']
			x, y_answers, df = extract_image_features(image_data['size'], image_files, image_filenames, image_data['data'], questions_df, word_to_vector, answers, vocab, question_seq_length)
			_, y_actions, _ = extract_action_word_caption_features(all_action_names, actions_df, image_files, image_filenames, image_captions, questions_df, word_to_vector, answers, vocab, max_seq_length, max_seq_length)
			y = [y_answers, y_actions[1]]
			model = action_vqa_model(image_data['size'], embedding_matrix, len(vocab), embedding_dim, question_seq_length, dropout_rate, len(answers), len(all_action_names))
		elif model_name == 'avg':
			x, y, df = extract_average_word_caption_features(image_files, image_filenames, image_captions, questions_df, word_to_vector, answers)
			model = average_text_model(embedding_dim, dropout_rate, len(answers))
		elif model_name == 'qc':
			x, y, df = extract_word_caption_features(image_files, image_filenames, image_captions, questions_df, word_to_vector, answers, vocab, question_seq_length, caption_seq_length)
			model = question_caption_model(embedding_matrix, len(vocab), embedding_dim, caption_seq_length, question_seq_length, dropout_rate, len(answers))
		elif model_name == 'qcatt':
			x, y, df = extract_attention_word_caption_features(image_files, image_filenames, image_captions, questions_df, word_to_vector, specific_answers, vocab, max_seq_length, max_seq_length)
			model = attention_question_caption_model(embedding_matrix, len(vocab), embedding_dim, max_seq_length, dropout_rate, len(specific_answers))
		elif model_name == 'qcact':
			x, y, df = extract_action_word_caption_features(all_action_names, actions_df, image_files, image_filenames, image_captions, questions_df, word_to_vector, answers, vocab, max_seq_length, max_seq_length)
			model = action_question_caption_model(embedding_matrix, len(vocab), embedding_dim, max_seq_length, dropout_rate, len(answers), len(all_action_names))
		elif model_name == 'bow':
			x, y, df = extract_bow_features(image_files, image_filenames, image_captions, questions_df, word_to_vector, answers, vocab)
			model = bow_model(len(vocab), dropout_rate, len(answers))
		elif model_name == 'qonly':
			x, y, df = extract_word_caption_features(image_files, image_filenames, image_captions, questions_df, word_to_vector, answers, vocab, question_seq_length, caption_seq_length)
			x = x[1]
			model = half_info_model(embedding_matrix, len(vocab), embedding_dim, question_seq_length, dropout_rate, len(answers))
		elif model_name == 'conly':
			x, y, df = extract_word_caption_features(image_files, image_filenames, image_captions, questions_df, word_to_vector, answers, vocab, question_seq_length, caption_seq_length)
			x = x[0]
			model = half_info_model(embedding_matrix, len(vocab), embedding_dim, caption_seq_length, dropout_rate, len(answers))
		elif model_name == 'qcimage':
			image_data = image_feature_data['resnet50']
			x, y, df = extract_image_features(image_data['size'], image_files, image_filenames, image_data['data'], questions_df, word_to_vector, answers, vocab, max_seq_length)
			x_captions, _, _ = extract_word_caption_features(image_files, image_filenames, image_captions, questions_df, word_to_vector, answers, vocab, max_seq_length, max_seq_length)
			x.append(x_captions[0])
			model = vqa_word_caption_model(image_data['size'], embedding_matrix, len(vocab), embedding_dim, max_seq_length, max_seq_length, dropout_rate, len(answers))

		return model, x, y, df

	for trial in range(0,trials):
		print 'Trial: [%d]' % (trial)
		shuffle(all_image_files)
		split = len(all_image_files)/3
		train_image_files = all_image_files[split:]
		test_image_files = all_image_files[:split]
		print '\tTrain image files count: [%d]' % (len(train_image_files))
		print '\tTest image files count: [%d]' % (len(test_image_files))

		for model_name in model_names:
			print 'Running %s model' % (model_name)
			model_weights_filename = os.path.join(results_dir, "%s-%d-model-final.h5" % (model_name, trial))
			ckpt_model_weights_filename = os.path.join(results_dir, "%s-%d-model-best.h5" % (model_name, trial))
			test_output_results_file = os.path.join(results_dir, "%s-%d-test_results.csv" % (model_name, trial))
			test_output_confusion_matrix_image = os.path.join(results_dir, "%s-%d-confusion_matrix.png" % (model_name, trial))

			callbacks = [ModelCheckpoint(filepath=ckpt_model_weights_filename, verbose=1, save_best_only=True), EarlyStopping(monitor='val_loss', patience=3, verbose=1)]

			print 'Loading training %s features...' % (model_name)
			model, x_train, y_train, train_df = create_model_and_data(model_name, train_image_files)
		
			if trial == 0:
				print model.summary()
			model.fit(x_train, y_train, epochs=number_of_epochs, batch_size=batch_size, shuffle="batch", verbose=1, validation_split=0.4, callbacks=callbacks)
			model.save_weights(model_weights_filename, overwrite=True)

			print 'Loading test %s features...' % (model_name)
			model, x_test, y_test, test_df = create_model_and_data(model_name, test_image_files)
			
			model.load_weights(ckpt_model_weights_filename)
			test_df['caption'] = test_df.apply(lambda x: image_captions[x['image_file']],axis=1)

			stats = {}
			stats['model'] = model_name
			stats['trial'] = trial

			if model_name == "qcatt":
				scores = model.evaluate(x_test, y_test, batch_size=batch_size)
				scores_map = dict(zip(model.metrics_names, scores))

				stats['reason_accuracy'] = scores_map['reason_output_acc']
				stats['relevance_accuracy'] = scores_map['relevance_output_acc']

				y_predict = model.predict(x_test)
				y_predict_answers = []
				y_true_answers = []
				y_predict_new = []
				y_true_new = []
				i = 0
				# print y_predict[0].shape
				# print y_predict[1].shape
				for y_val in y_predict[0]:
					y_predict_new.append(np.argmax(y_val))
					y_true_new.append(np.argmax(y_test[0][i]))
					y_predict_answers.append(answers.index_word[y_predict_new[i]])
					y_true_answers.append(answers.index_word[y_true_new[i]])
					i+=1

				y_predict_relevance = []
				for y_val in y_predict[1]:
					y_predict_relevance.append(y_val > 0.5)
				
				test_df['y_predict_relevance'] = y_predict_relevance
				test_df['y_true'] = y_true_answers
				test_df['y_predict'] = y_predict_answers
				test_df.to_csv(test_output_results_file)

				stats['accuracy'] = accuracy_score(y_true_new, y_predict_new)

			elif model_name in ["qcact", "imageact"]:
				scores = model.evaluate(x_test, y_test, batch_size=batch_size)
				scores_map = dict(zip(model.metrics_names, scores))

				stats['reason_accuracy'] = scores_map['reason_output_acc']
				stats['action_accuracy'] = scores_map['action_output_acc']

				y_predict = model.predict(x_test)
				y_predict_answers = []
				y_true_answers = []
				y_predict_new = []
				y_true_new = []
				i = 0
				# print y_predict[0].shape
				# print y_predict[1].shape
				for y_val in y_predict[0]:
					y_predict_new.append(np.argmax(y_val))
					y_true_new.append(np.argmax(y_test[0][i]))
					y_predict_answers.append(answers.index_word[y_predict_new[i]])
					y_true_answers.append(answers.index_word[y_true_new[i]])
					i+=1

				y_predict_actions = []
				for y_val in y_predict[1]:
					action = [all_action_names[i] for i,y in enumerate(y_val) if y > 0.5]
					y_predict_actions.append(','.join(action))

				test_df['y_predict_actions'] = y_predict_actions
				test_df['y_true'] = y_true_answers
				test_df['y_predict'] = y_predict_answers
				test_df.to_csv(test_output_results_file)

				stats['accuracy'] = accuracy_score(y_true_new, y_predict_new)

			else:
				y_predict = model.predict(x_test)
				y_predict_answers = []
				y_true_answers = []
				y_predict_new = []
				y_true_new = []
				i = 0
				for y_val in y_predict:
					y_predict_new.append(np.argmax(y_val))
					y_true_new.append(np.argmax(y_test[i]))
					y_predict_answers.append(answers.index_word[y_predict_new[i]])
					y_true_answers.append(answers.index_word[y_true_new[i]])
					i+=1
				test_df['y_true'] = y_true_answers
				test_df['y_predict'] = y_predict_answers
				test_df.to_csv(test_output_results_file)

				stats['accuracy'] = accuracy_score(y_true_new, y_predict_new)
				# stats['precision_macro'] = precision_score(y_true_new, y_predict_new, average='macro')
				# stats['precision_micro'] = precision_score(y_true_new, y_predict_new, average='micro')
				# stats['precision_weighted'] = precision_score(y_true_new, y_predict_new, average='weighted')
				# stats['recall_macro'] = recall_score(y_true_new, y_predict_new, average='macro')
				# stats['recall_micro'] = recall_score(y_true_new, y_predict_new, average='micro')
				# stats['recall_weighted'] = recall_score(y_true_new, y_predict_new, average='weighted')
				# stats['f1_macro'] = f1_score(y_true_new, y_predict_new, average='macro')
				# stats['f1_micro'] = f1_score(y_true_new, y_predict_new, average='micro')
				# stats['f1_weighted'] = f1_score(y_true_new, y_predict_new, average='weighted')
			print "Test accuracy: [%f]" % (stats['accuracy'])

			# new_labels = []
			# for a in answers.word_index:
			# 	old_index = answers.word_index[a]
			# 	if (old_index in y_true_new) or (old_index in y_predict_new):
			# 		new_labels.append(a)
			# 		new_index = len(new_labels)-1
			# 		for yi in range(len(y_true_new)):
			# 			if y_true_new[yi] == old_index:
			# 				y_true_new[yi] = new_index
			# 			if y_predict_new[yi] == old_index:
			# 				y_predict_new[yi] = new_index
			# cnf_matrix = confusion_matrix(y_true_new, y_predict_new)
			# np.set_printoptions(precision=2)

			# # Plot non-normalized confusion matrix
			# fig = plt.figure()
			# # plt.style.use('dark_background')
			# plot_confusion_matrix(cnf_matrix, classes=new_labels, title='Confusion matrix')
			# fig.savefig(test_output_confusion_matrix_image, facecolor=fig.get_facecolor())
			# plt.close(fig)
			# plt.show()

			print stats
			all_stats.append(stats)
			save_data(all_stats, output_stats_file)

	stats_df = save_data(all_stats, output_stats_file)
	print stats_df
	print 'Finished.'