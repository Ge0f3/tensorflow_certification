# Get data (10% of labels)
import zipfile
import wget
# Import the required modules for model creation
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
import tensorflow as tf
import tensorflow_hub as hub
import time


t1 = time.time()

#unzip the local file
# zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip", "r")
# zip_ref.extractall()
# zip_ref.close()

# Data loader

image_shape =(224,224)
batch_size = 32

train_dir = '10_food_classes_10_percent/train/'
test_dir = '10_food_classes_10_percent/test/'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(1/255.)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(1/255.)

train_data_10_percent = train_datagen.flow_from_directory(train_dir,
                                                          target_size=image_shape,
                                                          batch_size=batch_size,
                                                          class_mode='categorical',
                                                          shuffle=True)

test_data_10_percent = train_datagen.flow_from_directory(test_dir,
                                                         target_size=image_shape,
                                                         batch_size=batch_size,
                                                         class_mode='categorical')

# Setup data augmentation
data_augmentation = Sequential([
  preprocessing.RandomFlip("horizontal"), # randomly flip images on horizontal edge
  preprocessing.RandomRotation(0.2), # randomly rotate images by a specific amount
  preprocessing.RandomHeight(0.2), # randomly adjust the height of an image by a specific amount
  preprocessing.RandomWidth(0.2), # randomly adjust the width of an image by a specific amount
  preprocessing.RandomZoom(0.2), # randomly zoom into an image
  # preprocessing.Rescaling(1./255) # keep for models like ResNet50V2, remove for EfficientNet
], name="data_augmentation")

def create_model(model_url, num_classes):
    feature_extraction_lyaer = hub.KerasLayer(
        model_url,
        trainable=False,
        name='feature_extraction_layer',
        input_shape=image_shape + (3,)
    )

    model = tf.keras.Sequential([
        feature_extraction_lyaer,
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])

    return model

resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"





resnet_model =  create_model(resnet_url,train_data_10_percent.num_classes)


resnet_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics = ['accuracy']
)

resnet_model.fit(
    train_data_10_percent,
    epochs=5,
    validation_data = test_data_10_percent,
    validation_steps = len(test_data_10_percent)
)

t2 = time.time()

print(f"It takes {(t2-t1)/60}/")
