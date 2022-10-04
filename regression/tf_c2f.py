import tensorflow as tf
import numpy as np


#Create the dataset

celcuis_g = np.array( [-40,-10,0,8,15,22,38],dtype=float)
fahrenheit_g = np.array([celcuis*1.8+32 for celcuis in celcuis_g],dtype=float)

#Model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1,input_shape=celcuis_g[[0]].shape)
])
print(f"Model summary - {model.summary()} ")

#Compiling the model

model.compile(loss=tf.keras.losses.mean_squared_error,optimizer=tf.keras.optimizers.Adam(0.1))
print("Model Compiled")

history = model.fit(celcuis_g,fahrenheit_g,epochs=10)
print(f"The history - {history}")

print(f"The prediction for 1 - {model.predict([1])}")

model.save('f_to_C.h5')
print("Model Saved")

loaded_model = tf.keras.models.load_model('f_to_C.h5')
print("Model Loaded Successfully!")
print(f"The Loaded Model prediction for 1 - {loaded_model.predict([1])}")
