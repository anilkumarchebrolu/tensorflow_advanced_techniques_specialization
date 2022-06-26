import tensorflow as tf
import requests
import shutil
import matplotlib
from matplotlib import pyplot as plt
from imageio import mimsave
from PIL import Image
from IPython.display import display as display_fn

class Utilities:
    def __init__(self) -> None:
       pass
    
    def tensor_to_image(self, tensor):
        '''converts a tensor to an image'''
        tensor_shape = tf.shape(tensor)
        number_elem_shape = tf.shape(tensor_shape)
        if number_elem_shape > 3:
            assert tensor_shape[0] == 1
            tensor = tensor[0]
        return tf.keras.preprocessing.image.array_to_img(tensor) 


    def load_img(self, path_to_img):
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


    def load_images(self, content_path, style_path):
        '''loads the content and path images as tensors'''
        content_image = self.load_img("{}".format(content_path))
        style_image = self.load_img("{}".format(style_path))

        return content_image, style_image


    def imshow(self, image, title=None):
        '''displays an image with a corresponding title'''
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)
        if title:
            plt.title(title)
        
        
    def show_images_with_objects(self, images, titles=[]):
        '''displays a row of images with corresponding titles'''
        if len(images) != len(titles):
            return

        plt.figure(figsize=(20, 12))
        for idx, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(1, len(images), idx + 1)
            plt.xticks([])
            plt.yticks([])
            self.imshow(image, title)


    def display_gif(self, gif_path):
        '''displays the generated images as an animated gif'''
        with open(gif_path,'rb') as f:
            display_fn(Image(data=f.read(), format='png'))


    def create_gif(self, gif_path, images):
        '''creates animation of generated images'''
        mimsave(gif_path, images, fps=1)
        
        return gif_path


    def clip_image_values(self, image, min_value=0.0, max_value=255.0):
        '''clips the image pixel values by the given min and max'''
        return tf.clip_by_value(image, clip_value_min=min_value, clip_value_max=max_value)


    def preprocess_image(self, image):
        '''centers the pixel values of a given image to use with VGG-19'''
        image = tf.cast(image, dtype=tf.float32)
        image = tf.keras.applications.vgg19.preprocess_input(image)

        return image

    def save_image_from_url(self, image_path, url):
        response = requests.get(url, stream=True)
        with open(image_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response