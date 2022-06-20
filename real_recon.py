# %%
import numpy as np
import itertools
import pickle

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

EPOCH = 200
KERNEL_SIZE = 3
POOLING_SIZE = 2
BATCH_SIZE = 10

DATA_PATH = "./mit_data/"

def list_to_list(input_list):
    input_list_to_list = list(itertools.chain(*input_list))
    return input_list_to_list

# %%
# Dataload part
le = preprocessing.LabelEncoder()

record_list = []
pickle_input = dict()
X, y = [], []

print("[INFO] Read records file from ", DATA_PATH)
with open(DATA_PATH + 'RECORDS') as f:
    record_lines = f.readlines()

for i in range(len(record_lines)):
    record_list.append(str(record_lines[i].strip()))

for i in range(len(record_list)):
    temp_path = DATA_PATH + "mit" + record_list[i] + ".pkl"
    with open(temp_path, 'rb') as f:
        pickle_input = pickle.load(f)
        for i in range(len(pickle_input[0])):
            X.append(pickle_input[0][i])

        for i in range(len(pickle_input[1])):
            check_ann = pickle_input[1][i]
            temp_ann_list = list()
            if check_ann == "N":            # Normal
                temp_ann_list.append(0)

            elif check_ann == "S":          # Supra-ventricular
                temp_ann_list.append(1)

            elif check_ann == "V":          # Ventricular
                temp_ann_list.append(2)

            elif check_ann == "F":          # False alarm
                temp_ann_list.append(3)

            else:                           # Unclassed 
                temp_ann_list.append(4)
            y.append(temp_ann_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42, shuffle=True )

npx = np.array(X_train)
npy = np.array(y_train)
npx_vali = np.array(X_val)
npy_vali = np.array(y_val)
npx_test = np.array(X_test)
npy_test = np.array(y_test)

print("[SIZE]\t\tNpX lenght : {}\n\t\tNpY length : {}".format(npx.shape, npy.shape))
print("[SIZE]\t\tX_validation length : {}\n\t\ty_validation length : {}".format(npx_vali.shape, npy_vali.shape))
print("[SIZE]\t\tX_test length : {}\n\t\ty_test length : {}".format(npx_test.shape, npy_test.shape))

# %%
lefms = keras.Sequential([
    layers.Conv1D(32, 3, padding='same', input_shape=(428, 1)),
    layers.BatchNormalization(),
    layers.Activation(keras.activations.relu),

    layers.Conv1D(64, 3, padding='same'),
    layers.Conv1D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation(keras.activations.relu),
    layers.MaxPool1D(pool_size=(2), strides=2),
    
    layers.Conv1D(64, 3, padding='same'),
    layers.Conv1D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation(keras.activations.relu),
    layers.MaxPool1D(pool_size=(2), strides=2),
    
    layers.Conv1D(128, 3, padding='same'),
    layers.Conv1D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation(keras.activations.relu),
    layers.MaxPool1D(pool_size=(2), strides=2),
    
    layers.Conv1D(128, 3, padding='same'),
    layers.Conv1D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation(keras.activations.relu),
    layers.MaxPool1D(pool_size=(2), strides=2),
    
    layers.Conv1D(256, 3, padding='same'),
    layers.Conv1D(256, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation(keras.activations.relu),
    layers.MaxPool1D(pool_size=(2), strides=2),
    
    layers.Conv1D(256, 3, padding='same'),
    layers.Conv1D(256, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation(keras.activations.relu),
    layers.MaxPool1D(pool_size=(2), strides=2),

    layers.Reshape((1, 1536)),
    layers.Dropout(0.5),
    layers.GRU(1536),
    layers.Dense(192, activation='relu'),
    layers.Dense(5, activation='softmax')
])

# %%
lefms.summary()

# %%
lefms.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
)

# %%
# Train
X_train = tf.convert_to_tensor(npx)

y_train = tf.convert_to_tensor(npy)
y_train = tf.one_hot(y_train, 5)
y_train = tf.reshape(y_train, [75441, 5])
print("X_train shape : ",X_train.shape)
print("y_train shape",y_train.shape)

# Validation
X_vali = tf.convert_to_tensor(npx_vali)

y_vali = tf.convert_to_tensor(npy_vali)
y_vali = tf.one_hot(y_vali, 5)
y_vali = tf.reshape(y_vali, [12263, 5])
print("X_validation shape : ", X_vali.shape)
print("y_validation shape : ", y_vali.shape)

# Test
X_test = tf.convert_to_tensor(npx_test)

y_test = tf.convert_to_tensor(npy_test)
y_test = tf.one_hot(y_test, 5)
y_test = tf.reshape(y_test, [24895, 5])
print("X_test shape : ", X_test.shape)
print("y_test shape : ", y_test.shape)

# %%
with tf.device('/device:GPU:1'):
    lefms.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(X_vali, y_vali))

# %%
lefms.evaluate(X_test, y_test, batch_size=BATCH_SIZE)

# %%



