import tensorflow as tf
from utilities.utilities import Utilities
from utilities.neural_style_transfer_vgg19_model import NeuralStyleTransferVGG19Model
from utilities.style_content_feature_extraction import StyleContentFeatureExtraction
from utilities.style_content_loss import StyleContentLoss
from IPython.display import display as display_fn
from IPython.display import Image, clear_output
import numpy as np


def calculate_gradients(model, num_style_layers, num_content_layers, image, style_targets, content_targets, 
                        style_weight, content_weight, var_weight):
    """ Calculate the gradients of the loss with respect to the generated image
    Args:
        image: generated image
        style_targets: style features of the style image
        content_targets: content features of the content image
        style_weight: weight given to the style loss
        content_weight: weight given to the content loss
        var_weight: weight given to the total variation loss
    
    Returns:
        gradients: gradients of the loss with respect to the input image
    """
    
    with tf.GradientTape() as tape:
        
        # get the style image features
        style_features = StyleContentFeatureExtraction().get_style_image_features(model, image, num_style_layers) 
        
        # get the content image features
        content_features = StyleContentFeatureExtraction().get_content_image_features(model, image, num_style_layers) 
        
        # get the style and content loss
        loss = StyleContentLoss().get_style_content_loss(
            style_targets, style_features, content_targets, 
            content_features, style_weight, content_weight,
            num_style_layers, num_content_layers
        ) 

        # add the total variation loss 
        # explicit regularization term on the high frequency components of the image (lab2)
        loss += var_weight*tf.image.total_variation(image)

    # calculate gradients of loss with respect to the image
    gradients = tape.gradient(loss, image) 

    return gradients


def update_image_with_style(model, num_style_layers, num_content_layers, image, style_targets, content_targets, style_weight, 
                            var_weight, content_weight, optimizer):
    """
    Args:
        image: generated image
        style_targets: style features of the style image
        content_targets: content features of the content image
        style_weight: weight given to the style loss
        content_weight: weight given to the content loss
        var_weight: weight given to the total variation loss
        optimizer: optimizer for updating the input image
    """

    # calculate gradients using the function that you just defined.
    gradients = calculate_gradients(model, num_style_layers, num_content_layers, image, style_targets, content_targets, 
                                    style_weight, content_weight, var_weight) 

    # apply the gradients to the given image
    optimizer.apply_gradients([(gradients, image)]) 

    # clip the image using the utility clip_image_values() function
    image.assign(Utilities().clip_image_values(image, min_value=0.0, max_value=255.0))


def fit_style_transfer(vgg_model, num_style_layers, num_content_layers, style_image, content_image, style_weight=1e-2, content_weight=1e-4, 
                       var_weight=0, optimizer='adam', epochs=1, steps_per_epoch=1):
    """ Performs neural style transfer.
    Args:
        style_image: image to get style features from
        content_image: image to stylize 
        style_targets: style features of the style image
        content_targets: content features of the content image
        style_weight: weight given to the style loss
        content_weight: weight given to the content loss
        var_weight: weight given to the total variation loss
        optimizer: optimizer for updating the input image
        epochs: number of epochs
        steps_per_epoch = steps per epoch
    
    Returns:
        generated_image: generated image at final epoch
        images: collection of generated images per epoch  
    """

    images = []
    step = 0

    # get the style image features 
    style_targets = StyleContentFeatureExtraction().get_style_image_features(vgg_model, style_image, num_style_layers)
        
    # get the content image features
    content_targets = StyleContentFeatureExtraction().get_content_image_features(vgg_model, content_image, num_style_layers)

    # initialize the generated image for updates
    generated_image = tf.cast(content_image, dtype=tf.float32)
    generated_image = tf.Variable(generated_image) 
    
    # collect the image updates starting from the content image
    images.append(content_image)
    
    # incrementally update the content image with the style features
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            
            # Update the image with the style using the function that you defined
            update_image_with_style(vgg_model, num_style_layers, num_content_layers, generated_image, 
                                    style_targets, content_targets, style_weight,
                                    var_weight, content_weight, optimizer) 
            
            print(".", end='')

            if (m + 1) % 10 == 0:
                images.append(generated_image)
            
            # display the current stylized image
            clear_output(wait=True)
            display_image = Utilities().tensor_to_image(generated_image)
            display_fn(display_image)

            # append to the image collection for visualization later
            images.append(generated_image)
            print("Train step: {}".format(step))
    
    # convert to uint8 (expected dtype for images with pixels in the range [0,255])
    generated_image = tf.cast(generated_image, dtype=tf.uint8)

    return generated_image, images

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

    # Choose intermediate layers from the network(VGG19) to extract the style and content of the image:
    # For the style layers, you will use the first layer of each convolutional block.
    # For the content layer, you will use the second convolutional layer of the last convolutional block (just one layer)
    # style layers of interest
    style_layers = ['block1_conv1', 
                    'block2_conv1', 
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    # choose the content layer and put in a list
    content_layers = ['block5_conv2'] 

    # combine the two lists (put the style layers before the content layers)
    output_layers = style_layers + content_layers 

    # declare auxiliary variables holding the number of style and content layers
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    # Model
    vgg_model = NeuralStyleTransferVGG19Model().neural_style_transfer_vgg19_model(output_layers)
    vgg_model.summary()

    # define style and content weight
    style_weight =  2e-2
    content_weight = 1e-2 

    # define optimizer. learning rate decreases per epoch.
    adam = tf.optimizers.Adam(
        tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=20.0, decay_steps=100, decay_rate=0.50
        )
    )

    # start the neural style transfer
    stylized_image, display_images = fit_style_transfer(vgg_model, num_style_layers, num_content_layers, style_image=style_image, content_image=content_image, 
                                                        style_weight=style_weight, content_weight=content_weight,
                                                        var_weight=0, optimizer=adam, epochs=10, steps_per_epoch=100)

    # display GIF of Intermedite Outputs
    # GIF_PATH = 'outputs/style_transfer.gif' # lab1
    GIF_PATH = 'outputs/style_transfer_reg.gif' # lab2
    gif_images = [np.squeeze(image.numpy().astype(np.uint8), axis=0) for image in display_images]
    gif_path = Utilities().create_gif(GIF_PATH, gif_images)
    Utilities().display_gif(gif_path)