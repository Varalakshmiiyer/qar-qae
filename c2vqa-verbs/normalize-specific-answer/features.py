import numpy as np
from keras.preprocessing.sequence import pad_sequences

def extract_bow_features(image_files, image_filenames, image_captions, questions_df, word_to_vector, answers, vocab):
	x = []
	y_answers = []

	df = questions_df[questions_df['image_file'].isin(image_files)]

	for _,row in df.iterrows():
		feature = np.zeros(len(vocab))
		question_words = row['question'].split()
		caption_words = image_captions[row['image_file']].split()
		all_words = question_words + caption_words
		question_feature = []
		for word in all_words:
			if word in word_to_vector:
				feature[vocab(word)] += 1
			
		answer = np.zeros(len(answers.word_index))
		answer[answers(row['answer'])] = 1
		
		y_answers.append(answer)
		x.append(feature)

	return np.asarray(x),np.asarray(y_answers), df.copy()

def extract_average_word_caption_features(image_files, image_filenames, image_captions, questions_df, word_to_vector, answers):
	x = []
	y_answers = []

	df = questions_df[questions_df['image_file'].isin(image_files)]

	for _,row in df.iterrows():
		question_words = row['question'].split()
		question_feature = None
		question_word_count = 0
		for word in question_words:
			if word in word_to_vector:
				if question_feature is None:
					question_feature = np.asarray(word_to_vector[word])
				else:
					question_feature += np.asarray(word_to_vector[word])
				question_word_count += 1
		
		caption_words = image_captions[row['image_file']].split()
		caption_feature = None
		caption_word_count = 0
		for word in caption_words:
			if word in word_to_vector:
				if caption_feature is None:
					caption_feature = np.asarray(word_to_vector[word])
				else:
					caption_feature += np.asarray(word_to_vector[word])
				caption_word_count += 1
		
		# print caption_feature / float(caption_word_count)
		caption_feature=caption_feature / float(caption_word_count)
		question_feature=question_feature / float(question_word_count)

		# print caption_feature.shape
		# print question_feature.shape
		# print ''
		x.append(np.concatenate((caption_feature,question_feature),0))

		answer = np.zeros(len(answers.word_index))
		answer[answers(row['answer'])] = 1
		# print row['relevant']
		# print answer
		y_answers.append(answer)

	return np.asarray(x), np.asarray(y_answers), df.copy()

def extract_action_word_caption_features(all_action_names, actions_df, image_files, image_filenames, image_captions, questions_df, word_to_vector, answers, vocab, question_seq_length, caption_seq_length):
	x_captions = []
	x_questions = []
	y_answers = []
	y_actions = []

	df = questions_df[questions_df['image_file'].isin(image_files)]

	for _,row in df.iterrows():
		question_words = row['question'].split()
		question_feature = []
		for word in question_words:
			if word in word_to_vector:
				question_feature.append(vocab(word))
		x_questions.append(np.asarray(question_feature))

		caption_words = image_captions[row['image_file']].split()
		caption_feature = []
		for word in caption_words:
			if word in word_to_vector:
				caption_feature.append(vocab(word))
		x_captions.append(np.asarray(caption_feature))

		answer = np.zeros(len(answers.word_index))
		answer[answers(row['answer'])] = 1
		# print row['relevant']
		# print answer
		y_answers.append(answer)

		y_action = np.zeros(len(all_action_names))
		image_actions = row['image_actions'].split(',')
		image_actions.sort()

		for image_action in image_actions:
			y_action[all_action_names.index(image_action)] = 1
		y_actions.append(y_action)

	x_questions = pad_sequences(x_questions, question_seq_length)
	x_captions = pad_sequences(x_captions, caption_seq_length)
	return [np.asarray(x_captions), np.asarray(x_questions)], [np.asarray(y_answers),np.asarray(y_actions)], df.copy()

def extract_attention_word_caption_features(image_files, image_filenames, image_captions, questions_df, word_to_vector, answers, vocab, question_seq_length, caption_seq_length):
	x_captions = []
	x_questions = []
	y_answers = []
	y_relevant = []

	df = questions_df[questions_df['image_file'].isin(image_files)]

	for _,row in df.iterrows():
		question_words = row['question'].split()
		question_feature = []
		for word in question_words:
			if word in word_to_vector:
				question_feature.append(vocab(word))
		x_questions.append(np.asarray(question_feature))

		caption_words = image_captions[row['image_file']].split()
		caption_feature = []
		for word in caption_words:
			if word in word_to_vector:
				caption_feature.append(vocab(word))
		x_captions.append(np.asarray(caption_feature))

		answer = np.zeros(len(answers.word_index))
		answer[answers(row['specific_answer'])] = 1
		# print row['relevant']
		# print answer
		y_answers.append(answer)
		y_relevant.append(row['relevant'])

	x_questions = pad_sequences(x_questions, question_seq_length)
	x_captions = pad_sequences(x_captions, caption_seq_length)
	return [np.asarray(x_captions), np.asarray(x_questions)], [np.asarray(y_answers),np.asarray(y_relevant)], df.copy()

def extract_word_caption_features(image_files, image_filenames, image_captions, questions_df, word_to_vector, answers, vocab, question_seq_length, caption_seq_length):
	x_captions = []
	x_questions = []
	y_answers = []

	df = questions_df[questions_df['image_file'].isin(image_files)]

	for _,row in df.iterrows():
		question_words = row['question'].split()
		question_feature = []
		for word in question_words:
			if word in word_to_vector:
				question_feature.append(vocab(word))
		x_questions.append(np.asarray(question_feature))

		caption_words = image_captions[row['image_file']].split()
		caption_feature = []
		for word in caption_words:
			if word in word_to_vector:
				caption_feature.append(vocab(word))
		x_captions.append(np.asarray(caption_feature))

		answer = np.zeros(len(answers.word_index))
		answer[answers(row['answer'])] = 1
		# print row['relevant']
		# print answer
		y_answers.append(answer)

	x_questions = pad_sequences(x_questions, question_seq_length)
	x_captions = pad_sequences(x_captions, caption_seq_length)
	return [np.asarray(x_captions), np.asarray(x_questions)], np.asarray(y_answers), df.copy()

def extract_image_features(image_feature_size, image_files, image_filenames, image_features, questions_df, word_to_vector, answers, vocab, seq_length):
	x_images = []
	x_questions = []
	y_answers = []

 	# Normalizing images
	load_image_features = [image_features[image_filenames[image_file]] for image_file in image_files]
	tem = np.sqrt(np.sum(np.multiply(load_image_features, load_image_features), axis=1))
	load_image_features = np.divide(load_image_features, np.transpose(np.tile(tem,(image_feature_size,1))))

	df = questions_df[questions_df['image_file'].isin(image_files)]

	for _,row in df.iterrows():
		question_words = row['question'].split()
		question_feature = []
		for word in question_words:
			if word in word_to_vector:
				question_feature.append(vocab(word))
		x_questions.append(np.asarray(question_feature))

		x_images.append(load_image_features[image_files.index(row['image_file'])])
		answer = np.zeros(len(answers.word_index))
		answer[answers(row['answer'])] = 1
		# print row['relevant']
		# print answer
		y_answers.append(answer)

	x_questions = pad_sequences(x_questions, seq_length)
	return [np.asarray(x_images), np.asarray(x_questions)], np.asarray(y_answers), df.copy()