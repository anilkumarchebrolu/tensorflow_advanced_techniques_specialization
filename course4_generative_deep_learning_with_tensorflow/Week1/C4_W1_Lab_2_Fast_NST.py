import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from utilities.utilities import Utilities

def tensor_to_image(tensor):
    '''converts a tensor to an image'''
    tensor_shape = tf.shape(tensor)
    number_elem_shape = tf.shape(tensor_shape)
    if number_elem_shape > 3:
        assert tensor_shape[0] == 1
        tensor = tensor[0]
    return tf.keras.preprocessing.image.array_to_img(tensor) 


def load_img(path_to_img):
    '''loads an image as a tensor and scales it to 512 pixels'''
    max_dim = 512
    image = tf.io.read_file(path_to_img)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.shape(image)[:-1]
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    image = tf.image.convert_image_dtype(image, tf.uint8)

    return image


def load_images(content_path, style_path):
    '''loads the content and path images as tensors'''
    content_image = load_img("{}".format(content_path))
    style_image = load_img("{}".format(style_path))

    return content_image, style_image


def imshow(image, title=None):
    '''displays an image with a corresponding title'''
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)
    
    
def show_images_with_objects(images, titles=[]):
    '''displays a row of images with corresponding titles'''
    if len(images) != len(titles):
        return

    plt.figure(figsize=(20, 12))
    for idx, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), idx + 1)
        plt.xticks([])
        plt.yticks([])
        imshow(image, title)


if __name__ == '__main__':
    utilities = Utilities()

    image_names_paths_dict={
        "data/images/cafe.jpg": "https://cdn.pixabay.com/photo/2018/07/14/15/27/cafe-3537801_1280.jpg",
        "data/images/swan.jpg": "https://cdn.pixabay.com/photo/2017/02/28/23/00/swan-2107052_1280.jpg",
        "data/images/tnj.jpg": "https://i.dawn.com/large/2019/10/5db6a03a4c7e3.jpg",
        "data/images/rudolph.jpg": "https://cdn.pixabay.com/photo/2015/09/22/12/21/rudolph-951494_1280.jpg",
        "data/images/dynamite.jpg": "https://cdn.pixabay.com/photo/2015/10/13/02/59/animals-985500_1280.jpg",
        "data/images/painting.jpg": "https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
    }

    for image_path, image_url in image_names_paths_dict.items():
        utilities.save_image_from_url(image_path, image_url)

    # set default content and style image
    content_path = "data/images/swan.jpg"
    style_path = "data/images/painting.jpg"

    # Show loaded images
    # display the content and style image
    content_image, style_image = utilities.load_images(content_path, style_path)
    utilities.show_images_with_objects([content_image, style_image], 
                            titles=[f'content image: {content_path}',
                                    f'style image: {style_path}'])

    # this will take a few minutes to load
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # stylize the image using the model you just downloaded
    stylized_image = hub_module(tf.image.convert_image_dtype(content_image, tf.float32), 
                                tf.image.convert_image_dtype(style_image, tf.float32))[0]

    # convert the tensor to image
    final_image = tensor_to_image(stylized_image)
    final_image.save("outputs/fast_nst.png")