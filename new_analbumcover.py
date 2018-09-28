from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('album_covers/train',
                                                    target_size=(150, 150),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory('album_covers/validation',
                                                        target_size=(150, 150),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

model.fit_generator(train_generator,
                    steps_per_epoch=2000 // batch_size,
                    epochs=20,
                    validation_data=validation_generator,
                    validation_steps=800 // batch_size)

# model.save_weights('first_try_weights.h5')
model.save('first_try_model.h5')

# model1 = load_model('first_try_model.h5')
#
#
#
# img1 = load_img('album_covers/metal/DiamondHead_DiamondHead_HeavyMetal.jpg', target_size=(150, 150))
# arr1 = img_to_array(img1)
#
# img2 = load_img('album_covers/metal/Eisbrecher_Schock_IndustrialMetal.jpg', target_size=(150,150))
# arr2 = img_to_array(img2)
#
# imgs = np.array([arr1, arr2])
#
# model1.predict_classes(imgs)

