from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import itertools
import pickle

EPOCH = 200
KERNEL_SIZE = 3
POOLING_SIZE = 2
BATCH_SIZE = 10

DATA_PATH = "./mit_data/"

def list_to_list(input_list):
    input_list_to_list = list(itertools.chain(*input_list))
    return input_list_to_list

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

X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

print("[SIZE]\t\tX length : {}\n\t\ty length : {}".format(len(X), len(y)))
print("[SIZE]\t\tX_test length : {}\n\t\ty_test length : {}".format(len(X_test), len(y_test)))

npx = np.array(X)
npy = np.array(y)
npx_test = np.array(X_test)
npy_test = np.array(y_test)

print("[SIZE]\t\tNpX lenght : {}\n\t\tNpY length : {}".format(npx.shape, npy.shape))
print("[SIZE]\t\tX_test length : {}\n\t\ty_test length : {}".format(npx_test.shape, npy_test.shape))

X = tf.cast(tf.constant(X), dtype=tf.float32)
# y = tf.cast(tf.constant(y), dtype=tf.string)
X_test = tf.cast(tf.constant(X_test), dtype=tf.float32)
# y_test = tf.cast(tf.constant(y_test), dtype=tf.string)

print(npx.shape)

X_np = np.array(X)
y_np = np.array(y)

tf.random.set_seed(42)
input_size = layers.Input(shape=X_np.shape)

lefms = keras.Sequential([
    layers.Conv1D(428, 32, padding='same', input_shape=(input_size)),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Conv1D(186, 32, padding='same', input_shape=(input_size)),
    layers.Conv1D(186, 32, padding='same', input_shape=(input_size)),
    layers.BatchNormalization(32),
    layers.Activation(keras.activations.relu),
    layers.MaxPool2D(pool_size=(93, 32)),
    
    layers.Conv1D(93, 32, padding='same', input_shape=(input_size)),
    layers.Conv1D(93, 64, padding='same', input_shape=(input_size)),
    layers.BatchNormalization(64),
    layers.Activation(keras.activations.relu),
    layers.MaxPool2D(pool_size=(47, 64)),
    
    layers.Conv1D(47, 64, padding='same', input_shape=(input_size)),
    layers.Conv1D(47, 64, padding='same', input_shape=(input_size)),
    layers.BatchNormalization(64),
    layers.Activation(keras.activations.relu),
    layers.MaxPool2D(pool_size=(24, 128)),
    
    layers.Conv1D(12, 128, padding='same', input_shape=(input_size)),
    layers.Conv1D(12, 128, padding='same', input_shape=(input_size)),
    layers.BatchNormalization(128),
    layers.Activation(keras.activations.relu),
    layers.MaxPool2D(pool_size=(12, 128)),
    
    layers.Conv1D(6, 256, padding='same', input_shape=(input_size)),
    layers.Conv1D(6, 256, padding='same', input_shape=(input_size)),
    layers.BatchNormalization(256),
    layers.Activation(keras.activations.relu),
    layers.MaxPool2D(pool_size=(6, 256)),
    
    layers.Conv1D(6, 256, padding='same', input_shape=(input_size)),
    layers.Conv1D(6, 256, padding='same', input_shape=(input_size)),
    layers.BatchNormalization(256),
    layers.Activation(keras.activations.relu),
    layers.MaxPool2D(pool_size=(3, 256)),

    layers.Dropout(1),
    layers.GRU(768, input_shape=(input_size)),
    layers.Dense(96),
    layers.Softmax(axis=5)
])

print(lefms)
print(len(lefms.layers))

lefms.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
)

X_np = tf.convert_to_tensor(X_np)
y_np = tf.convert_to_tensor(y_np)

lefms.fit(
        # tf.expand_dims(X_np, axis=-1),
        x = X_np,
        y = y_np,
        epochs = EPOCH,
        batch_size = BATCH_SIZE
)

lefms.fit(X_np, y_np, batch_size=BATCH_SIZE, epochs=EPOCH)


