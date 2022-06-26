import tensorflow as tf
from utilities.utilities import Utilities

class StyleContentFeatureExtraction:
    def __init__(self) -> None:
        self.utitlites = Utilities()

    def gram_matrix(self, input_tensor):
        """ Calculates the gram matrix and divides by the number of locations
        Args:
            input_tensor: tensor of shape (batch, height, width, channels)
            
        Returns:
            scaled_gram: gram matrix divided by the number of locations
        """

        # calculate the gram matrix of the input tensor
        gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) 

        # get the height and width of the input tensor
        input_shape = tf.shape(input_tensor) 
        height = input_shape[1] 
        width = input_shape[2] 

        # get the number of locations (height times width), and cast it as a tf.float32
        num_locations = tf.cast(height * width, tf.float32)

        # scale the gram matrix by dividing by the number of locations
        scaled_gram = gram / num_locations
            
        return scaled_gram
    
    def get_style_image_features(self, model, image, num_style_layers):  
        """ Get the style image features
        
        Args:
            image: an input image
            
        Returns:
            gram_style_features: the style features as gram matrices
        """
        # preprocess the image using the given preprocessing function
        preprocessed_style_image = self.utitlites.preprocess_image(image) 

        # get the outputs from the custom vgg model that you created using vgg_model()
        outputs = model(preprocessed_style_image) 

        # Get just the style feature layers (exclude the content layer)
        style_outputs = outputs[:num_style_layers] 

        # for each style layer, calculate the gram matrix for that layer and store these results in a list
        gram_style_features = [self.gram_matrix(style_layer) for style_layer in style_outputs] 

        return gram_style_features
    
    def get_content_image_features(self, model, image, num_style_layers):
        """ Get the content image features
        
        Args:
            image: an input image
            
        Returns:
            content_outputs: the content features of the image
        """
        # preprocess the image
        preprocessed_content_image = self.utitlites.preprocess_image(image)
            
        # get the outputs from the vgg model
        outputs = model(preprocessed_content_image) 

        # get the content layers of the outputs
        content_outputs = outputs[num_style_layers:]

        # return the content layer outputs of the content image
        return content_outputs