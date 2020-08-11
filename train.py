import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPool2D



classifier = keras.Sequential()
classifier.add(Conv2D(32,(3,3),input_shape = (64,64,1),activation="relu"))
classifier.add(Dropout(0.25))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))
classifier.add(Conv2D(32,(3,3),))

classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=2, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory('Data',
                                                 target_size=(64, 64),
                                                 batch_size=2,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch=10,
        epochs=10,
        )

model_json = classifier.to_json()
with open("model_json","w") as json_file:
    json_file.write(model_json)

classifier.save_weights("model.h5")
print("Saved Model to Disk")