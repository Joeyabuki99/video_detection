import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from tensorflow import keras
from keras.applications import EfficientNetB3
from keras import layers
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

ROOT_DIR_YOLO = "/user/sfasulo/dataset/detector_dataset/dataset_other"
ROOT_DIR_CLASS = "/user/sfasulo/dataset/classifier_dataset/"


IMG_SIZE = 300  
BATCH_SIZE = 256
NUM_CLASSES = 3

dataset_class = os.path.join(ROOT_DIR_CLASS)
contents = os.listdir(dataset_class)

class_labels = []

for item in contents:
  all_classes = os.listdir(ROOT_DIR_CLASS + '/' + item)
  for i in all_classes:
    class_labels.append((item, str('dataset_path' + '/' + item) + '/' + i))

# Build a dataframe
df = pd.DataFrame(data=class_labels, columns=['Labels', 'image'])

dataset_path = os.listdir(ROOT_DIR_CLASS)

images = []
labels = []

# Resize the images
for i in dataset_path:
    data_path = ROOT_DIR_CLASS + str(i)
    filenames = [i for i in os.listdir(data_path) ]

    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(i)

# Normalization phase
images = np.array(images)
images = images.astype('float32') / 255.0
images.shape

# Label encoding
y=df['Labels'].values
y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)

y=y.reshape(-1,1)

ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y) #.toarray()

images, Y = shuffle(images, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=415)

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    model = EfficientNetB3(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

model = build_model(num_classes=NUM_CLASSES)

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    #plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

#plot_hist(hist)

preds = model.evaluate(test_x, test_y)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

epochs = 30
#hist = model.fit(train_x, train_y, epochs=epochs, verbose=2)
hist = model.fit(train_x, train_y, batch_size = BATCH_SIZE, epochs = epochs, verbose = 2)
plot_hist(hist)

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

unfreeze_model(model)

epochs = 4  # @param {type: "slider", min:4, max:10}
hist = model.fit(train_x, train_y, batch_size = BATCH_SIZE, epochs = 5, verbose = 2)
plot_hist(hist)

model.evaluate(test_x, test_y)

# Salva il modello
# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#transfer-learning-from-pretrained-weights
model.save('/user/sfasulo/saved_models/efficientnet_model_unfreeze_256.h5')
