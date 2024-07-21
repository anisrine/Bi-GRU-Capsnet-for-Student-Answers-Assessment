import pandas as pd
import re
from sklearn.model_selection import train_test_split
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D,Dot,Lambda,BatchNormalization
from keras.models import Model
import numpy as np
from vendor.Capsule.Capsule_Keras import *
from keras.layers import Concatenate
import itertools
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
import gensim.downloader as api



data = pd.read_csv('AnswersData.csv', error_bad_lines=False)
data_training=data[['StudentAnswer','ReferenceAnswers','Assessment']]
data_test = pd.read_csv('test_data_regression.csv', error_bad_lines=False)
data_test = data_test[['StudentAnswer','ReferenceAnswers']]

print("check nan values")
print (data.isnull().any())

print (data_test.isnull().any())

#tokenization and cleaning
def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()
    return text

vocab = {}
inverse_vocabulary = ['<test>']
#word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

answer_cols=['StudentAnswer','ReferenceAnswers']

for dataset in [data_training,data_test ]:
    for index, row in dataset.iterrows():
        for answer in answer_cols:
           answer_to_numb = []
           for word in text_to_word_list(row[answer]):
               if word not in vocab:
                   vocab[word] = len(inverse_vocabulary)
                   answer_to_numb.append(len(inverse_vocabulary))
                   inverse_vocabulary.append(word)
               else:
                   answer_to_numb.append(vocab[word])

           dataset.set_value(index,answer, answer_to_numb)

embed_size = 300
model = api.load("word2vec-google-news-300")  # download the model and return as object ready for use
word_vectors = model.wv
embedding_matrix = np.zeros((len(vocab) + 1, embed_size))

for word, i in vocab.items():
      try:
           embedding_vector = word_vectors[word]
           embedding_matrix[i] = embedding_vector
      except KeyError:
          embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), embed_size)

del(word_vectors)

maxlen_1 = max(data_training.StudentAnswer.map(lambda x: len(x)).max(),
                     data_training.ReferenceAnswers.map(lambda x: len(x)).max(),
                     data_test.StudentAnswer.map(lambda x: len(x)).max(),
                     data_test.ReferenceAnswers.map(lambda x: len(x)).max())

X = data_training[answer_cols]
Y = data_training['Assessment']



X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2)

X_train = {'left': X_train.StudentAnswer, 'right': X_train.ReferenceAnswers}
X_validation = {'left': X_validation.StudentAnswer, 'right': X_validation.ReferenceAnswers}
X_test = {'left': data_test.StudentAnswer, 'right': data_test.ReferenceAnswers}

Y_train = np_utils.to_categorical(Y_train, num_classes=4)  # Y_train.value
Y_validation = np_utils.to_categorical(Y_validation, num_classes=4)  # Y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation, X_test], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=maxlen_1)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)
#

gru_len = 256
Routings = 3
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.9
rate_drop_dense = 0.28
embed_size = 300


def get_model():
    input1 = Input(shape=(maxlen_1,))
    embed_layer_sa = Embedding(len(embedding_matrix),
                            embed_size,
                            weights = [embedding_matrix],
                            input_length=maxlen_1,
                             trainable=False)(input1)
    embed_layer_sa = SpatialDropout1D(rate_drop_dense)(embed_layer_sa)

    x = Bidirectional(GRU(gru_len,
                          activation ='relu',
                          dropout=dropout_p,
                          recurrent_dropout=dropout_p,
                          return_sequences=True))(embed_layer_sa)


    capsule = Capsule(
        num_capsule=Num_capsule,
        dim_capsule=Dim_capsule,
        routings=Routings,
        share_weights=True)(x)

    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    capsule = LeakyReLU()(capsule)
    capsule = BatchNormalization()(capsule)

    x = Flatten()(capsule)

    input2 = Input(shape=(maxlen_1,))
    embed_layer_ra = Embedding(len(embedding_matrix),
                            embed_size,
                            weights = [embedding_matrix],
                            input_length=maxlen_1,
                             trainable=False)(input2)
    embed_layer_ra = SpatialDropout1D(rate_drop_dense)(embed_layer_ra)

    y = Bidirectional(GRU(gru_len,
                        activation='relu',
                        dropout=dropout_p,
                        recurrent_dropout=dropout_p,
                        return_sequences=True))(embed_layer_ra)

    capsule_2 = Capsule(
          num_capsule=Num_capsule,
          dim_capsule=Dim_capsule,
          share_weights=True)(y)

    capsule_2 = Flatten()(capsule_2)
    capsule_2 = Dropout(dropout_p)(capsule_2)
    capsule_2 = LeakyReLU()(capsule_2)
    capsule_2 = BatchNormalization()(capsule_2)

    y = Flatten()(capsule_2)

    merged = Concatenate()([x, y])

    output = Dense(4, activation='softmax')(merged)

    model = Model(inputs=[input1,input2], outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer= 'adam',
        metrics=['accuracy'])

    return model

def main():

    model = get_model()

    #Hyperpapermeters
    batch_size =64
    epochs = 40

    model.fit( [X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=epochs,
               validation_data=([X_validation['left'], X_validation['right']], Y_validation) )

    prediction = model.predict([X_test['left'],X_test['right']],batch_size=64, verbose=1 )

    score = model.evaluate([X_validation['left'], X_validation['right']], Y_validation, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
