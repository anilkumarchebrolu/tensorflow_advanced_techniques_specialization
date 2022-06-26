import tensorflow as tf
import numpy as np
import PIL
import tensorflow_datasets as tfds

class MnistObjectDetectionDataset:
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size

    def draw_bounding_boxes_on_image_array(self, image,
                                        boxes,
                                        color=[],
                                        thickness=1,
                                        display_str_list=()):
        """Draws bounding boxes on image (numpy array).
        Args:
            image: a numpy array object.
            boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
                The coordinates are in normalized format between [0, 1].
            color: color to draw bounding box. Default is red.
            thickness: line thickness. Default value is 4.
            display_str_list_list: a list of strings for each bounding box.
        Raises:
            ValueError: if boxes is not a [N, 4] array
        """
        image_pil = PIL.Image.fromarray(image)
        rgbimg = PIL.Image.new("RGBA", image_pil.size)
        rgbimg.paste(image_pil)
        self.draw_bounding_boxes_on_image(rgbimg, boxes, color, thickness,
                                    display_str_list)
        return np.array(rgbimg)
    

    def draw_bounding_boxes_on_image(self,
                                    image,
                                    boxes,
                                    color=[],
                                    thickness=1,
                                    display_str_list=()):
        """Draws bounding boxes on image.
        Args:
            image: a PIL.Image object.
            boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
                The coordinates are in normalized format between [0, 1].
            color: color to draw bounding box. Default is red.
            thickness: line thickness. Default value is 4.
            display_str_list: a list of strings for each bounding box.
                                
        Raises:
            ValueError: if boxes is not a [N, 4] array
        """
        boxes_shape = boxes.shape
        if not boxes_shape:
            return
        if len(boxes_shape) != 2 or boxes_shape[1] != 4:
            raise ValueError('Input must be of size [N, 4]')
        for i in range(boxes_shape[0]):
            self.draw_bounding_box_on_image(image, boxes[i, 1], boxes[i, 0], boxes[i, 3],
                                    boxes[i, 2], color[i], thickness, display_str_list[i])
            
    def draw_bounding_box_on_image(self,
                                image,
                                ymin,
                                xmin,
                                ymax,
                                xmax,
                                color='red',
                                thickness=1,
                                display_str=None,
                                use_normalized_coordinates=True):
        """Adds a bounding box to an image.
        Bounding box coordinates can be specified in either absolute (pixel) or
        normalized coordinates by setting the use_normalized_coordinates argument.
        Args:
            image: a PIL.Image object.
            ymin: ymin of bounding box.
            xmin: xmin of bounding box.
            ymax: ymax of bounding box.
            xmax: xmax of bounding box.
            color: color to draw bounding box. Default is red.
            thickness: line thickness. Default value is 4.
            display_str_list: string to display in box
            use_normalized_coordinates: If True (default), treat coordinates
            ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
            coordinates as absolute.
        """
        draw = PIL.ImageDraw.Draw(image)
        im_width, im_height = image.size
        if use_normalized_coordinates:
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                        ymin * im_height, ymax * im_height)
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line([(left, top), (left, bottom), (right, bottom),
                    (right, top), (left, top)], width=thickness, fill=color)

    
    def read_image_tfds(self, image, label):
        '''
        Transforms each image in dataset by pasting it on a 75x75 canvas at random locations.
        '''
        xmin = tf.random.uniform((), 0 , 48, dtype=tf.int32)
        ymin = tf.random.uniform((), 0 , 48, dtype=tf.int32)
        image = tf.reshape(image, (28,28,1,))
        image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
        image = tf.cast(image, tf.float32)/255.0
        xmin = tf.cast(xmin, tf.float32)
        ymin = tf.cast(ymin, tf.float32)
    
        xmax = (xmin + 28) / 75
        ymax = (ymin + 28) / 75
        xmin = xmin / 75
        ymin = ymin / 75
        return image, (tf.one_hot(label, 10), [xmin, ymin, xmax, ymax])

    def get_training_dataset(self):
        '''
        Loads and maps the training split of the dataset using the map function. Note that we try to load the gcs version since TPU can only work with datasets on Google Cloud Storage.
        '''
        dataset = tfds.load("mnist", split="train", as_supervised=True, try_gcs=True)
        dataset = dataset.map(self.read_image_tfds, num_parallel_calls=16)
        dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
        dataset = dataset.repeat() # Mandatory for Keras for now
        dataset = dataset.batch(self.batch_size, drop_remainder=True) # drop_remainder is important on TPU, batch size must be fixed
        dataset = dataset.prefetch(-1)  # fetch next batches while training on the current one (-1: autotune prefetch buffer size)
        return dataset
    
    def get_validation_dataset(self):
        '''
        Loads and maps the validation split of the dataset using the map function. Note that we try to load the gcs version since TPU can only work with datasets on Google Cloud Storage.
        '''  
        dataset = tfds.load("mnist", split="test", as_supervised=True, try_gcs=True)
        dataset = dataset.map(self.read_image_tfds, num_parallel_calls=16)

        #dataset = dataset.cache() # this small dataset can be entirely cached in RAM
        dataset = dataset.batch(10000, drop_remainder=True) # 10000 items in eval dataset, all in one batch
        dataset = dataset.repeat() # Mandatory for Keras for now
        return dataset

    def dataset_to_numpy_util(self, training_dataset, validation_dataset, N):
        # pull a batch from the datasets. This code is not very nice, it gets much better in eager mode (TODO)        
        # get one batch from each: 10000 validation digits, N training digits
        batch_train_ds = training_dataset.unbatch().batch(N)
        
        # eager execution: loop through datasets normally
        if tf.executing_eagerly():
            for validation_digits, (validation_labels, validation_bboxes) in validation_dataset:
                validation_digits = validation_digits.numpy()
                validation_labels = validation_labels.numpy()
                validation_bboxes = validation_bboxes.numpy()
                break
            
            for training_digits, (training_labels, training_bboxes) in batch_train_ds:
                training_digits = training_digits.numpy()
                training_labels = training_labels.numpy()
                training_bboxes = training_bboxes.numpy()
                break
        
        # these were one-hot encoded in the dataset
        validation_labels = np.argmax(validation_labels, axis=1)
        training_labels = np.argmax(training_labels, axis=1)
        
        return (training_digits, training_labels, training_bboxes,
                validation_digits, validation_labels, validation_bboxes)