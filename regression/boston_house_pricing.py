import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=0.3, seed=42
)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model_1 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(x_train.shape[1], name="Input_Layer"),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1),
    ]
)


model_1.compile(
    loss=tf.keras.losses.mean_absolute_error, optimizer=tf.keras.optimizers.Adam(0.01)
)


history_model_1 = model_1.fit(x_train, y_train, epochs=50, verbose=1)

print("Model Predictions !!!")
for i in range(10):
    print(
        model_1.predict(np.array(x_test[i].reshape(13,), ndmin=2)).tolist()[
            0
        ][0],
        y_test[i],
    )

print(f"Model Summary \n {model_1.summary()}")


print("Plotting the loss metric")
pd.DataFrame(history_model_1.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()


print(model_1.evaluate(x_test, y_test))

model_1.save("boston_house_pricing.h5")


loaded_model = tf.keras.models.load_model("boston_house_pricing.h5")

print(
    f"Model Evaluation result of the loaded model - {model_1.evaluate(x_test,y_test)}"
)

print(f"Model Size {os.stat('boston_house_pricing.h5').st_size/1000}kb")
