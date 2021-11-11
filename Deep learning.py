import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Input
from keras.models import Model
from keras.layers import ZeroPadding1D, BatchNormalization, Conv1D, Activation, Dense, \
    GlobalAveragePooling1D, MaxPooling1D, Add
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

#Input 12-lead ECG Data (Reference data type: pickle)
for i in range(n): #n: number of files(w
    with open('/"User_directory"/"file_name".pickle', 'rb') as f:
        globals()['Data_{}'.format(i)] = pickle.load(f)

Input_ECG = np.concatenate([Data_n.... ], axis=0) #n: number of files

# Input of ECG signal Data
# Shape of the 12-lead Signal [Case(n), Length(5000), leads(12)]
Input_ECG = Input_ECG[:, :, ["lead1", "lead2", "v2"]] # Three ECG signal (lead1, lead2, v2)
input_tensor = Input(shape=(5000, 3), dtype='float32', name='input')

results = list()


#Target Data
y = pd.read_excel('/"User_directory"/"File_name".xlsx', index_col = 0)
Y = np.array(y.iloc[:,["Target"]]) #Select of label


# Definition of layers of ResNet
def conv1_layer(x):
    x = ZeroPadding1D(padding=3)(x)
    x = Conv1D(64, 3, strides=1, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding1D(padding=1)(x)

    return x


def conv2_layer(x):
    x = MaxPooling1D(2)(x)

    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv1D(64, 1, strides=1, padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(64, 1, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(256, 1, strides=1, padding='valid')(x)
            shortcut = Conv1D(256, 1, strides=1, padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv1D(64, 1, strides=1, padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(64, 3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(256, 1, strides=1, padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv3_layer(x):
    shortcut = x

    for i in range(4):
        if (i == 0):
            x = Conv1D(128, 1, strides=2, padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(128, 3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(512, 1, strides=1, padding='valid')(x)
            shortcut = Conv1D(512, 1, strides=2, padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv1D(128, 1, strides=1, padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(128, 3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(512, 1, strides=1, padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv4_layer(x):
    shortcut = x

    for i in range(6):
        if (i == 0):
            x = Conv1D(256, 1, strides=2, padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(256, 3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(1024, 1, strides=1, padding='valid')(x)
            shortcut = Conv1D(1024, 1, strides=2, padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv1D(256, 1, strides=1, padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(256, 3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(1024, 1, strides=1, padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv5_layer(x):
    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv1D(512, 1, strides=2, padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(512, 3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(2048, 1, strides=1, padding='valid')(x)
            shortcut = Conv1D(2048, 1, strides=2, padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv1D(512, 1, strides=1, padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(512, 3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv1D(2048, 1, strides=1, padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x

#Classification model
x = conv1_layer(input_tensor)
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
x = conv5_layer(x)

x = GlobalAveragePooling1D()(x)
output_tensor = Dense(1, activation='sigmoid')(x)
resnet = Model(input_tensor, output_tensor)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
cvscores = []
cvloss = []

seed = 0
x_train_all, X_test, y_train_all, y_test = train_test_split(Input_ECG, Y, test_size=0.1,random_state=seed)


#Model training(10-fold)
i = 0
for train_index, valid_index in kf.split(x_train_all, y_train_all):
    i = i + 1
    X_train, X_valid = x_train_all[train_index], x_train_all[valid_index]
    y_train, y_valid = y_train_all[train_index], y_train_all[valid_index]

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    resnet = Model(input_tensor, output_tensor)
    resnet.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    resnet.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, callbacks=[callback])

    scores = resnet.evaluate(X_test, y_test, verbose=0)
    y_pred = resnet.predict(X_test)

    resnet.save('"User_directory"/"File_name"_{}.h5'.format(i))#Save model

    resnet_acc = y_pred.round()

    globals()['Report_{}'.format(i)] = classification_report(y_test, resnet_acc, output_dict=True)
    df_obs = pd.DataFrame(globals()['Report_{}'.format(i)])
    df_obs.to_csv('"User_directory"/"File_name"_{}.csv'.format(i), index=False)#Save classification report

    print("%s: %.2f%%" % (resnet.metrics_names[1], scores[1] * 100))
    print(classification_report(y_test, resnet_acc))

    cvscores.append(scores[1] * 100)
    cvloss.append(scores[0])

df_cvscore = pd.DataFrame(cvscores)
df_loss = pd.DataFrame(cvloss)
df_cvscore.to_csv('"User_directory"/cvsscores.csv', index=False)
df_loss.to_csv('"User_directory"/_cvloss.csv', index=False)