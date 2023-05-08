import os

# 기본 경로
base_dir = '../../open/Codes/basic/'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_furniture_dir = os.path.join(train_dir, '가구수정')
train_skirting_dir = os.path.join(train_dir, '걸레받이수정')
print(train_furniture_dir)
print(train_skirting_dir)

validation_furniture_dir = os.path.join(validation_dir, '가구수정')
validation_skirting_dir = os.path.join(validation_dir, '걸레받이수정')
print(train_furniture_dir)
print(train_skirting_dir)

train_furniture_fnames = os.listdir( train_furniture_dir )
train_skirting_fnames = os.listdir( train_skirting_dir )
print(train_furniture_fnames[:5])
print(train_skirting_fnames[:5])

print('Total training furniture images :', len(os.listdir(train_furniture_dir)))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

nrows, ncols = 4, 4
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols*3, nrows*3)

pic_index+=8

next_cat_pix = [os.path.join(train_furniture_dir, fname)
                for fname in train_furniture_fnames[ pic_index-8:pic_index]]

next_dog_pix = [os.path.join(train_skirting_dir, fname)
                for fname in train_skirting_fnames[ pic_index-8:pic_index]]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

import tensorflow as tf


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

#model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics = ['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                  batch_size=9,
                                                  class_mode='binary',
                                                  target_size=(150, 150))
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                       batch_size=9,
                                                       class_mode  = 'binary',
                                                       target_size = (150, 150))

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=8,
                    epochs=10,
                    validation_steps=9,
                    verbose=2)

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()