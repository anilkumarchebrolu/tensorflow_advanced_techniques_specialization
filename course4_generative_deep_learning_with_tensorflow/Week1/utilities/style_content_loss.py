import tensorflow as tf

class StyleContentLoss:
    def get_style_loss(self, features, targets):
        """Expects two images of dimension h, w, c
        
        Args:
            features: tensor with shape: (height, width, channels)
            targets: tensor with shape: (height, width, channels)

        Returns:
            style loss (scalar)
        """
        # get the average of the squared errors
        style_loss = tf.reduce_mean(tf.square(features - targets))
        return style_loss
    
    def get_content_loss(self, features, targets):
        """Expects two images of dimension h, w, c
        
        Args:
            features: tensor with shape: (height, width, channels)
            targets: tensor with shape: (height, width, channels)
        
        Returns:
            content loss (scalar)
        """
        # get the sum of the squared error multiplied by a scaling factor
        content_loss = 0.5 * tf.reduce_sum(tf.square(features - targets))
            
        return content_loss

    def get_style_content_loss(self, style_targets, style_outputs, content_targets, 
                           content_outputs, style_weight, content_weight,
                           num_style_layers, num_content_layers):
        """ Combine the style and content loss
        
        Args:
            style_targets: style features of the style image
            style_outputs: style features of the generated image
            content_targets: content features of the content image
            content_outputs: content features of the generated image
            style_weight: weight given to the style loss
            content_weight: weight given to the content loss

        Returns:
            total_loss: the combined style and content loss

        """
        style_loss = tf.add_n(
            [self.get_style_loss(style_output, style_target)
            for style_output, style_target in zip(style_outputs, style_targets)]
        )

        content_loss = tf.add_n(
            [self.get_content_loss(content_output, content_target)
            for content_output, content_target in zip(content_outputs, content_targets)]
        )

         # scale the style loss by multiplying by the style weight and dividing by the number of style layers
        style_loss = style_loss * style_weight / num_style_layers 

        # scale the content loss by multiplying by the content weight and dividing by the number of content layers
        content_loss = content_loss * content_weight / num_content_layers 
            
        # sum up the style and content losses
        total_loss = style_loss + content_loss 

        return total_loss
