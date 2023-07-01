import tensorflow as tf
import matplotlib.pyplot as plt
(train_images, train_labels),(test_images, test_label)=tf.keras.datasets.mnist.load_data()
train_images=train_images/255.0
test_images=test_images/255.0
print(train_images.shape)
print(test_images.shape)
print(train_labels)


plt.imshow(train_images[0],cmap='gray')
plt.show()
mod = tf.keras.models.Sequential()
mod.add(tf.keras.layers.Flatten(input_shape=(28,28)))
mod.add(tf.keras.layers.Dense(128, activation='relu'))
mod.add(tf.keras.layers.Dense(10, activation='softmax'))

mod.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
mod.fit(train_images,train_labels,epochs=3)

val_loss, val_acc = mod.evaluate(test_images,test_label)
print("accuracy",val_acc)

mod.save('basic model')