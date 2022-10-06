import tensorflow as tf
import time


t1 = time.time()
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()

print((x_train.shape,y_train.shape,x_test.shape,y_test.shape))

x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
    tf.keras.layers.Conv2D(32,3,activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32,3,activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation='softmax')
])

print("Compiling Model")
model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

print("Training Model")
model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))

print("Evaluating the model")
print(model.evaluate(x_test,y_test))

print("saving the model")
model.save('mnist.h5')

t2 = time.time()

print(f"It takes {(t2-t1)/60}/")
