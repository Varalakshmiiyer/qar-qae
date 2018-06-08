from keras.models import *
from keras.layers import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import h5py

# https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/

def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print "Creating text model..."
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=seq_length, trainable=False))
    model.add(LSTM(units=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=512, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1024, activation='tanh'))
    return model

def joint_text_model(embedding_matrix, num_words, embedding_dim, caption_seq_length, question_seq_length, dropout_rate):
    print "Creating joint text model..."

    encoder_a = Sequential()
    encoder_a.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=question_seq_length, trainable=False))
    encoder_a.add(LSTM(units=512, return_sequences=True, input_shape=(caption_seq_length, embedding_dim)))
    encoder_a.add(Dropout(dropout_rate))
    encoder_a.add(LSTM(units=512, return_sequences=True))

    encoder_b = Sequential()
    encoder_b.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=caption_seq_length, trainable=False))
    encoder_b.add(LSTM(units=512, return_sequences=True, input_shape=(caption_seq_length, embedding_dim)))
    encoder_b.add(Dropout(dropout_rate))
    encoder_b.add(LSTM(units=512, return_sequences=True))

    encoder_c = Sequential()
    encoder_c.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=1, trainable=False))
    encoder_c.add(Dense(512, input_dim=embedding_dim))
    encoder_c.add(Reshape((512,)))

    decoder = Sequential()
    decoder.add(Merge([encoder_a, encoder_b, encoder_c], mode='concat'))
    decoder.add(Dropout(dropout_rate))
    decoder.add(LSTM(units=512, return_sequences=False))
    decoder.add(Dropout(dropout_rate))
    decoder.add(Dense(1024, activation='tanh'))
    return decoder

def img_model(image_feature_size, dropout_rate):
    print "Creating image model..."
    model = Sequential()
    model.add(Dense(1024, input_dim=image_feature_size, activation='tanh'))
    return model

def vqa_model(image_feature_size, embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    vgg_model = img_model(image_feature_size, dropout_rate)
    lstm_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    encoder_c = Sequential()
    encoder_c.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=1, trainable=False))
    encoder_c.add(Dense(1024, input_dim=embedding_dim, activation='tanh'))
    encoder_c.add(Reshape((1024,)))
    print "Merging final model..."
    fc_model = Sequential()
    fc_model.add(Merge([vgg_model, lstm_model, encoder_c], mode='mul'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(1000, activation='tanh'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return fc_model

def vqa_word_caption_model(image_feature_size, embedding_matrix, num_words, embedding_dim, caption_seq_length, question_seq_length, dropout_rate, num_classes):
    vgg_model = img_model(image_feature_size, dropout_rate)
    lstm_question_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, question_seq_length, dropout_rate)
    lstm_caption_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, caption_seq_length, dropout_rate)
    encoder_c = Sequential()
    encoder_c.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=1, trainable=False))
    encoder_c.add(Dense(1024, input_dim=embedding_dim, activation='tanh'))
    encoder_c.add(Reshape((1024,)))
    print "Merging final joint text image model..."
    fc_model = Sequential()
    fc_model.add(Merge([vgg_model, lstm_question_model, lstm_caption_model, encoder_c], mode='mul'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(1000, activation='tanh'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return fc_model

def half_info_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    decoder = Sequential()
    decoder.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length= seq_length, trainable=False))
    decoder.add(Bidirectional(LSTM(200), input_shape=(seq_length,embedding_dim)))
    decoder.add(Dense(150, activation='relu'))
    decoder.add(Dense(100, activation='relu'))
    decoder.add(Dropout(dropout_rate))
    decoder.add(Dense(num_classes, activation='softmax'))
    decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return decoder

def action_question_caption_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes, num_actions):
    emb = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)
    lstm = Bidirectional(LSTM(200), input_shape=(seq_length,embedding_dim))
    c_input = Input(shape=(seq_length,))
    q_input = Input(shape=(seq_length,))
    e_input = Input(shape=(1,))

    encoder_c = emb(c_input)
    encoder_c = lstm(encoder_c)

    encoder_q = emb(q_input)
    encoder_q = lstm(encoder_q)

    encoder_e = emb(e_input)
    encoder_e = Dense(400)(encoder_e)
    encoder_e = Reshape((400,))(encoder_e)

    merged_output = merge([encoder_c, encoder_q, encoder_e], mode='concat')
    output1 = Dense(200, activation='relu')(merged_output)
    output = Dense(200, activation='relu')(output1)
    output = Dropout(dropout_rate)(output)

    reason_output = Dense(num_classes, activation='softmax', name='reason_output')(output)

    action_output = Dense(num_actions, activation='relu', name='action_output')(encoder_c)

    decoder = Model(inputs=[c_input, q_input, e_input], outputs=[reason_output, action_output])
    decoder.compile(optimizer='rmsprop', metrics=['accuracy'], loss={'reason_output': 'categorical_crossentropy', 'action_output': 'mse'},
              loss_weights={'reason_output': 1., 'action_output': 0.5})
    return decoder

def question_caption_model(embedding_matrix, num_words, embedding_dim, caption_seq_length, question_seq_length, dropout_rate, num_classes):
    encoder_a = Sequential()
    encoder_a.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=caption_seq_length, trainable=False))
    encoder_a.add(Bidirectional(LSTM(200), input_shape=(caption_seq_length,embedding_dim)))

    encoder_b = Sequential()
    encoder_b.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=question_seq_length, trainable=False))
    encoder_b.add(Bidirectional(LSTM(200), input_shape=(question_seq_length,embedding_dim)))

    encoder_c = Sequential()
    encoder_c.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=1, trainable=False))
    encoder_c.add(Dense(400, input_dim=embedding_dim))
    encoder_c.add(Reshape((400,)))

    decoder = Sequential()
    decoder.add(Merge([encoder_a, encoder_b, encoder_c], mode='concat'))
    decoder.add(Dense(150, activation='relu'))
    decoder.add(Dense(100, activation='relu'))
    decoder.add(Dropout(dropout_rate))
    decoder.add(Dense(num_classes, activation='softmax'))
    decoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return decoder

def average_text_model(embedding_dim, dropout_rate, num_classes):
    decoder = Sequential()
    decoder.add(Dense(150, input_dim=embedding_dim*2, activation='relu'))
    decoder.add(Dense(125, activation='relu'))
    decoder.add(Dense(100, activation='relu'))
    decoder.add(Dropout(dropout_rate))
    decoder.add(Dense(num_classes, activation='softmax'))
    decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return decoder

def bow_model(num_words, dropout_rate, num_classes):
    decoder = Sequential()
    decoder.add(Dense(300, input_dim=num_words, activation='relu'))
    decoder.add(Dense(200, activation='relu'))
    decoder.add(Dense(150, activation='relu'))
    decoder.add(Dropout(dropout_rate))
    decoder.add(Dense(num_classes, activation='softmax'))
    decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return decoder