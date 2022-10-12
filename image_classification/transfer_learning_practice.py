# Get data (10% of labels)
import zipfile
import wget
import tensorflow as tf

zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip", "r")
zip_ref.extractall()
zip_ref.close()

# Data loader

train_datagen  = tf.keras.preprocessing.ImageDataGenerator(1/255.)
test_datagen = tf.keras.preprocessing.ImageDataGenerator(1/255.)