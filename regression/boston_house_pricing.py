import tensorflow as tf


(x_train,y_train),(x_test,y_test)=tf.keras.datasets.boston_housing.load_data(
    path='boston_housing.npz',test_split=0.3,seed=42)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_1.compile(
    loss=  tf.keras.losses.mean_absolute_error,
    optimizer=tf.keras.optimizers.Adam(0.01)
)

history_model_1 = model_1.fit(
    x_train,y_train,epochs=50,verbose=1
)


print(model_1.evaluate(x_test,y_test))

model_1.save('boston_house_pricing.h5')

loaded_model = tf.keras.models.load_model('boston_house_pricing.h5')

print(f"Model Evaluation result of the loaded model - {model_1.evaluate(x_test,y_test)}")

