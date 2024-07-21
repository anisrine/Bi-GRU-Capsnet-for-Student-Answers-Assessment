import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D,Reshape,BatchNormalization,Add
from keras.models import Model
from vendor.Capsule.Capsule_Keras import *
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from keras.utils import np_utils
import tensorflow_hub as hb
from keras.layers.core import Lambda
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Read data
data = pd.read_csv('training_txt.csv')
data_training=data[['StudentAnswer','ReferenceAnswers','Assessment']]
data_test = pd.read_csv('testing_txt.csv')
data_test = data_test[['StudentAnswer','ReferenceAnswers']]

print("check nan values")
print (data.isnull().any())
print (data_test.isnull().any())

answer_cols=['StudentAnswer','ReferenceAnswers']

X = data_training[answer_cols]
Y = data_training['Assessment']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2)

X_train = {'left': X_train.StudentAnswer, 'right': X_train.ReferenceAnswers}
X_validation = {'left': X_validation.StudentAnswer, 'right': X_validation.ReferenceAnswers}
X_test = {'left': data_test.StudentAnswer, 'right': data_test.ReferenceAnswers}

X_train['left'] = X_train['left'].tolist()
X_train['left'] = [' '.join(t.split()[0:150]) for t in X_train['left']]
X_train['left'] = np.array(X_train['left'], dtype=object)[:, np.newaxis]

X_train['right'] = X_train['right'].tolist()
X_train['right'] = [' '.join(t.split()[0:150]) for t in X_train['right']]
X_train['right'] = np.array(X_train['right'], dtype=object)[:, np.newaxis]

X_validation['left'] = X_validation['left'].tolist()
X_validation['left'] = [' '.join(t.split()[0:150]) for t in X_validation['left']]
X_validation['left'] = np.array(X_validation['left'], dtype=object)[:, np.newaxis]

X_validation['right'] = X_validation['right'].tolist()
X_validation['right'] = [' '.join(t.split()[0:150]) for t in X_validation['right']]
X_validation['right'] = np.array(X_validation['right'], dtype=object)[:, np.newaxis]


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

Y_train = labelencoder.fit_transform(Y_train)
Y_validation=labelencoder.fit_transform(Y_validation)

Y_train = np_utils.to_categorical(Y_train, num_classes=4)  # Y_train.value
Y_validation = np_utils.to_categorical(Y_validation, num_classes=4)  # Y_validation.values

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


gru_len = 1024
Routings = 3
Num_capsule = 15
Dim_capsule = 16
dropout_p = 0.9
rate_drop_dense = 0.28
embed_size = 100

# Reduce TensorFlow logging output.
tf.logging.set_verbosity(tf.logging.ERROR)
url = "https://tfhub.dev/google/elmo/2"
embed = hb.Module(url)


def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]


# Initialize session
sess = tf.Session()
K.set_session(sess)

K.set_learning_phase(1)

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

def expand_dims(x):
    return K.expand_dims(x, 1)

def expand_dims_output_shape(input_shape):
    return (input_shape[0], 1, input_shape[1])

def get_model():
    input1 = Input(shape=(None,),dtype=tf.string)
    embed_layer_sa = Lambda(ELMoEmbedding, output_shape=(None,1024))(input1)
    embed_layer_sa = Lambda(expand_dims)(embed_layer_sa)
    embed_layer_sa = SpatialDropout1D(rate_drop_dense)(embed_layer_sa)

    print("shape :", embed_layer_sa.shape)
    x = Bidirectional(GRU(gru_len,
                          dropout=dropout_p,
                          recurrent_dropout=dropout_p,
                          return_sequences=True))(embed_layer_sa)


    capsule_1 = Capsule(
        num_capsule=Num_capsule,
        dim_capsule=Dim_capsule,
        routings=Routings,
        share_weights=True)(x)

    capsule_1 = Flatten()(capsule_1)
    capsule_1 = Dropout(dropout_p)(capsule_1)
    capsule_1 = LeakyReLU()(capsule_1)
    capsule_1 = BatchNormalization()(capsule_1)

    x = Flatten()(capsule_1)

    input2 = Input(shape=(None,),dtype=tf.string)
    embed_layer_ra = Lambda(ELMoEmbedding, output_shape=(None,1024 ))(input2)
    embed_layer_ra = Lambda(expand_dims)(embed_layer_ra)
    embed_layer_ra = SpatialDropout1D(rate_drop_dense)(embed_layer_ra)

    y = Bidirectional(GRU(gru_len,
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

    y = Flatten()( capsule_2)

    merged = Concatenate()([x, y])

    output =  Dense(2, activation='softmax')(merged)

    model = Model(inputs=[input1,input2], outputs=output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer= 'adam',
        metrics=['accuracy'])

    model.summary()

    return model

def main():

    model = get_model()
    batch_size =64
    epochs = 200

    print("new type", type(X_train['left']))

    model.fit( [X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=epochs)
    score = model.evaluate([X_validation['left'], X_validation['right']], Y_validation, verbose=0)

    print('Test Accuracy:', score[1])

    prediction = model.predict([X_test['left'], X_test['right']], batch_size=1, verbose=1)
    y_preds = np.argmax(prediction, axis=-1)

    y_preds.astype(np.float32).reshape((-1, 1))

    y_true = np.array(data_test['Assessment'])
    y_true.astype(np.float32).reshape((-1, 1))

    #Output the confusion matrix
    print(confusion_matrix(y_true, y_preds))




if __name__ == '__main__':
    main()
