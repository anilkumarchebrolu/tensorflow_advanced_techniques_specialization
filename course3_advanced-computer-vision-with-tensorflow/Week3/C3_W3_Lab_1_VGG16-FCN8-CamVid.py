import tensorflow as tf
import numpy as np
import os
from vgg_fcn8_model import VGGFCN8Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy


BATCH_SIZE = 8

# Utilities for preparing the datasets
def get_dataset_slice_paths(image_dir, label_map_dir):
    '''
    generates the lists of image and label map paths
    
    Args:
        image_dir (string) -- path to the input images directory
        label_map_dir (string) -- path to the label map directory

    Returns:
        image_paths (list of strings) -- paths to each image file
        label_map_paths (list of strings) -- paths to each label map
    '''
    image_file_list = os.listdir(image_dir)
    label_map_file_list = os.listdir(label_map_dir)
    image_paths = [os.path.join(image_dir, fname) for fname in image_file_list]
    label_map_paths = [os.path.join(label_map_dir, fname) for fname in label_map_file_list]

    return image_paths, label_map_paths


def get_training_dataset(image_paths, label_map_paths):
    '''
    Prepares shuffled batches of the training set.
    
    Args:
        image_paths (list of strings) -- paths to each image file in the train set
        label_map_paths (list of strings) -- paths to each label map in the train set

    Returns:
        tf Dataset containing the preprocessed train set
    '''
    training_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    training_dataset = training_dataset.map(map_filename_to_image_and_mask)
    training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
    training_dataset = training_dataset.batch(BATCH_SIZE)
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.prefetch(-1)

    return training_dataset


def get_validation_dataset(image_paths, label_map_paths):
    '''
    Prepares batches of the validation set.
    
    Args:
        image_paths (list of strings) -- paths to each image file in the val set
        label_map_paths (list of strings) -- paths to each label map in the val set

    Returns:
        tf Dataset containing the preprocessed validation set
    '''
    validation_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    validation_dataset = validation_dataset.map(map_filename_to_image_and_mask)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.repeat()  

    return validation_dataset


def map_filename_to_image_and_mask(t_filename, a_filename, height=224, width=224):
    '''
    Preprocesses the dataset by:
        * resizing the input image and label maps
        * normalizing the input image pixels
        * reshaping the label maps from (height, width, 1) to (height, width, 12)

    Args:
        t_filename (string) -- path to the raw input image
        a_filename (string) -- path to the raw annotation (label map) file
        height (int) -- height in pixels to resize to
        width (int) -- width in pixels to resize to

    Returns:
        image (tensor) -- preprocessed image
        annotation (tensor) -- preprocessed annotation
    '''

    # Convert image and mask files to tensors 
    img_raw = tf.io.read_file(t_filename)
    anno_raw = tf.io.read_file(a_filename)
    image = tf.image.decode_jpeg(img_raw)
    annotation = tf.image.decode_jpeg(anno_raw)
    
    # Resize image and segmentation mask
    image = tf.image.resize(image, (height, width,))
    annotation = tf.image.resize(annotation, (height, width,))
    image = tf.reshape(image, (height, width, 3,))
    annotation = tf.cast(annotation, dtype=tf.int32)
    annotation = tf.reshape(annotation, (height, width, 1,))
    stack_list = []

    # Reshape segmentation masks
    for c in range(len(class_names)):
        mask = tf.equal(annotation[:,:,0], tf.constant(c))
        stack_list.append(tf.cast(mask, dtype=tf.int32))
    
    annotation = tf.stack(stack_list, axis=2)

    # Normalize pixels in the input image
    image = image/127.5
    image -= 1

    return image, annotation

def get_images_and_segments_test_arrays():
    '''
    Gets a subsample of the val set as your test set

    Returns:
        Test set containing ground truth images and label maps
    '''
    y_true_segments = []
    y_true_images = []
    test_count = 64

    ds = validation_dataset.unbatch()
    ds = ds.batch(101)

    for image, annotation in ds.take(1):
        y_true_images = image
        y_true_segments = annotation


    y_true_segments = y_true_segments[:test_count, : ,: , :]
    y_true_segments = np.argmax(y_true_segments, axis=3)  

    return y_true_images, y_true_segments

def compute_metrics(y_true, y_pred):
    '''
    Computes IOU and Dice Score.

    Args:
        y_true (tensor) - ground truth label map
        y_pred (tensor) - predicted label map
    '''
    
    class_wise_iou = []
    class_wise_dice_score = []

    smoothening_factor = 0.00001

    for i in range(12):
        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area
        
        iou = (intersection + smoothening_factor) / (combined_area - intersection + smoothening_factor)
        class_wise_iou.append(iou)
        
        dice_score =  2 * ((intersection + smoothening_factor) / (combined_area + smoothening_factor))
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score

if __name__ == "__main__":
    # dataset source url
    # https://drive.google.com/uc?id=0B0d9ZiqAgFkiOHR1NTJhWVJMNEU

    # pixel labels in the video frames
    class_names = ['sky', 'building','column/pole', 'road', 'side walk', 'vegetation',
                   'traffic light', 'fence', 'vehicle', 'pedestrian', 'byciclist', 'void']

    dataset_path = "/d_drive/Anil/personal/learning/TensorFlow_Advanced_Techniques_Specialization_by_deeplearning.ai/tensorflow_advanced_techniques_specialization/datasets/course_3_week3_fcnn_data/dataset1/dataset1"                    

    # get the paths to the images
    training_image_paths, training_label_map_paths = get_dataset_slice_paths(os.path.join(dataset_path, 'images_prepped_train'), os.path.join(dataset_path, 'annotations_prepped_train'))
    validation_image_paths, validation_label_map_paths = get_dataset_slice_paths(os.path.join(dataset_path, 'images_prepped_test'), os.path.join(dataset_path, 'annotations_prepped_test'))

    # generate the train and val sets
    training_dataset = get_training_dataset(training_image_paths, training_label_map_paths)
    validation_dataset = get_validation_dataset(validation_image_paths, validation_label_map_paths)

    # Model creation
    inputs = Input(shape=(224,224,3,))
    num_classes = 12
    fcn_model = VGGFCN8Model(inputs, num_classes).final_model()

    # Compile the model
    sgd = SGD(lr=1E-2, momentum=0.9, nesterov=True)
    loss = CategoricalCrossentropy()
    fcn_model.compile(
        optimizer=sgd,
        loss=loss,
        metrics=['accuracy']
    )

    ## Train the model
    # number of training images
    train_count = 367

    # number of validation images
    validation_count = 101

    EPOCHS = 170

    steps_per_epoch = train_count//BATCH_SIZE
    validation_steps = validation_count//BATCH_SIZE

    history = fcn_model.fit(training_dataset, steps_per_epoch=steps_per_epoch, 
                        validation_data=validation_dataset, validation_steps=validation_steps, 
                        epochs=EPOCHS)
    print(history)


    # load the ground truth images and segmentation masks
    y_true_images, y_true_segments = get_images_and_segments_test_arrays()

    # get the model prediction
    results = fcn_model.predict(validation_dataset, steps=validation_steps)

    # for each pixel, get the slice number which has the highest probability
    results = np.argmax(results, axis=3)

    # input a number from 0 to 63 to pick an image from the test set
    integer_slider = 0

    # compute metrics
    iou, dice_score = compute_metrics(y_true_segments[integer_slider], results[integer_slider])  

    # compute class-wise metrics
    cls_wise_iou, cls_wise_dice_score = compute_metrics(y_true_segments, results)

    # print IOU for each class
    for idx, iou in enumerate(cls_wise_iou):
        spaces = ' ' * (13-len(class_names[idx]) + 2)
        print("{}{}{} ".format(class_names[idx], spaces, iou)) 
    
    # print the dice score for each class
    for idx, dice_score in enumerate(cls_wise_dice_score):
        spaces = ' ' * (13-len(class_names[idx]) + 2)
        print("{}{}{} ".format(class_names[idx], spaces, dice_score)) 


