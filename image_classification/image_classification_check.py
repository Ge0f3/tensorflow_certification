import tensorflow as tf



(x_train,y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()

print((x_train.shape,y_train.shape,x_test.shape,y_test.shape))

