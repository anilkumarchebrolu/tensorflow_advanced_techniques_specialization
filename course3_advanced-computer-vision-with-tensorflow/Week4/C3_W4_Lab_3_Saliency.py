# Saliency maps
# Like class activation maps, saliency maps also tells us what parts of the image the model is focusing on when making its predictions.

# The main difference is in saliency maps, we are just shown the relevant pixels instead of the learned features.
# You can generate saliency maps by getting the gradient of the loss with respect to the image pixels.
# This means that changes in certain pixels that strongly affect the loss will be shown brightly in your saliency map.

import tensorflow as tf
import tensorflow_hub as hub
import requests
import shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt

def save_image_from_url(url, image_path):
    response = requests.get(url, stream=True)
    with open(image_path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response

def load_and_normalize_image(image_path):
    image = cv2.imread(image_path) # read image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # change from bgr to rgb
    image = cv2.resize(image, (300, 300)) / 255.0 # resize and normalize
    image = np.expand_dims(image, axis=0) # adding batch dimension
    return image

def compute_gradients(image, model, expected_output):
    # Computing gradients of the loss with respect to the input image pixels.
    with tf.GradientTape() as tape:
        inputs = tf.cast(image, dtype=tf.float32) # cast to float
        tape.watch(inputs) # watch the input pixels
        predictions = model(inputs) # generate the predictions

        # get the loss
        loss = tf.keras.losses.categorical_crossentropy(
            expected_output, predictions
        )
    
        gradients = tape.gradient(loss, inputs) # get the gradients w.r.t. to the inputs
    return gradients

def save_saliency_maps(normalized_tensor, image):
    plt.figure(figsize=(32, 32))
    plt.axis('off')
    plt.imshow(normalized_tensor, cmap='gray')
    plt.imsave("results/model_focus_areas.png", normalized_tensor)

    gradient_color = cv2.applyColorMap(normalized_tensor.numpy(), cv2.COLORMAP_HOT)
    gradient_color = gradient_color / 255.0
    image = np.squeeze(image)
    super_imposed = cv2.addWeighted(image, 0.5, gradient_color, 0.5, 0.0)

    plt.figure(figsize=(32, 32))
    plt.imshow(super_imposed)
    plt.axis('off')
    plt.imsave("results/super_imposed.png", super_imposed)


def generate_saliency_maps(gradients, image):
    # by uusing generated gradients, 
    # we will do some postprocessing to generate the saliency maps 
    # and overlay it on the image

    # reduce the RGB image to grayscale
    grayscale_tensor = tf.reduce_sum(tf.abs(gradients), axis=-1)

    # normalize the pixel values to be in the range [0, 255].
    # the max value in the grayscale tensor will be pushed to 255.
    # the min value will be pushed to 0.
    normalized_tensor = tf.cast(
        255
        * (grayscale_tensor - tf.reduce_min(grayscale_tensor))
        / (tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor)),
        tf.uint8,
    )

    # remove the channel dimension to make the tensor a 2d tensor
    normalized_tensor = tf.squeeze(normalized_tensor)

    ##Sanity check to see the results of conversion
    # max and min value in the grayscale tensor
    print(np.max(grayscale_tensor[0]))
    print(np.min(grayscale_tensor[0]))
    print()

    # coordinates of the first pixel where the max and min values are located
    max_pixel = np.unravel_index(np.argmax(grayscale_tensor[0]), grayscale_tensor[0].shape)
    min_pixel = np.unravel_index(np.argmin(grayscale_tensor[0]), grayscale_tensor[0].shape)
    print(max_pixel)
    print(min_pixel)
    print()

    # these coordinates should have the max (255) and min (0) value in the normalized tensor
    print(normalized_tensor[max_pixel])
    print(normalized_tensor[min_pixel]) 

    save_saliency_maps(normalized_tensor, image)

if __name__ == '__main__':
    # For the classifier, you will use the Inception V3 model available in Tensorflow Hub. 
    # This has pre-trained weights that is able to detect 1001 classes. 
    # grab the model from Tensorflow hub and append a softmax activation
    model = tf.keras.Sequential([
        hub.KerasLayer('https://tfhub.dev/google/tf2-preview/inception_v3/classification/4'),
        tf.keras.layers.Activation('softmax')
    ])

    # build the model based on a specified batch input shape
    model.build([None, 300, 300, 3])

    # You will download a photo of a Siberian Husky that our model will classify.
    url = "https://cdn.pixabay.com/photo/2018/02/27/14/11/the-pacific-ocean-3185553_960_720.jpg"
    image_path = "downloads/husky.jpg"
    save_image_from_url(url, image_path)

    # Loading and normalizing the image to pass into the model.
    image = load_and_normalize_image(image_path)

    # Siberian Husky's class ID in ImageNet
    class_index = 251
    
    # convert to one hot representation to match our softmax activation in the model definition
    num_classes = 1001 # number of classes in the model's training data
    expected_output = tf.one_hot([class_index] * image.shape[0], num_classes)

    # Compute gradients
    gradients = compute_gradients(image, model, expected_output)
    generate_saliency_maps(gradients, image)




    