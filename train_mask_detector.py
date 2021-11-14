from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras .layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import os

ini_LR=1e-4
epochs=20
batchsize=32


DIRECTORY=r"C:\Users\John Joel\PycharmProjects\face_mask_detection\dataset"
CATEGORIES=['with_mask','without_mask']

print("Loading Images")
data=[]
labels=[]

for category in CATEGORIES:
    path=os.path.join(DIRECTORY,category)
    for img in os.listdir(path):
        img_path=os.path.join(path,img)
        image=load_img(img_path,target_size=(224,224))
        image=img_to_array(image)
        image=preprocess_input(image)

        data.append(image)
        labels.append(category)

lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

data=np.array(data,dtype='float32')
labels=np.array(labels)

X_train,X_test,Y_train,Y_test=train_test_split(data,labels,test_size=0.2,stratify=labels,random_state=40)

aug=ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

basemodel=MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

headmodel=basemodel.output
headmodel=AveragePooling2D(pool_size=(7,7))(headmodel)
headmodel=Flatten(name="flatten")(headmodel)
headmodel=Dense(128,activation="relu")(headmodel)
headmodel=Dropout(0.5)(headmodel)
headmodel=Dense(2,activation="softmax")(headmodel)

model=Model(inputs=basemodel.input,outputs=headmodel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process

for layer in basemodel.layers:
    layer.trainable=False

print("compiling model")
opt=Adam(lr=ini_LR,decay=ini_LR/epochs)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

print("training model")
H=model.fit(
    aug.flow(X_train,Y_train,batch_size=batchsize),
    steps_per_epoch=len(X_train)//batchsize,
    validation_data=(X_test,Y_test),
    validation_steps=len(X_test)//batchsize,
    epochs=epochs
)

print('evaluating model')
predics=model.predict(X_test,batch_size=batchsize)
predics=np.argmax(predics,axis=1)

print(classification_report(Y_test.argmax(axis=1),predics,target_names=lb.classes_))

model.save("mask_detector.model",save_format="h5")

N=epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")