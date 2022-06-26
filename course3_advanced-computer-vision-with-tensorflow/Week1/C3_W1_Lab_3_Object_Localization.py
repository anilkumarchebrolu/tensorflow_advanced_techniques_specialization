import tensorflow as tf
from utils.mnist_object_detection_dataset import MnistObjectDetectionDataset
from metrics.intersection_over_union import IntersectionOverUnion
import numpy as np


class ObjectLocalizationModel:
    def __init__(self, classes, bounding_box_shape=4) -> None:
        self.classes = classes
        self.bounding_box_shape = bounding_box_shape

    def feature_extractor(self, inputs):
        '''
        Feature extractor is the CNN that is made up of convolution and pooling layers.
        '''
        x = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=3, input_shape=(75, 75, 1))(inputs)
        x = tf.keras.layers.AveragePooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(32,kernel_size=3, activation='relu')(x)
        x = tf.keras.layers.AveragePooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(64,kernel_size=3, activation='relu')(x)
        x = tf.keras.layers.AveragePooling2D((2, 2))(x)
        return x

        
    def dense_layers(self, inputs):
        '''
        dense_layers adds a flatten and dense layer.
        This will follow the feature extraction layers
        '''

        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        return x

    
    def classifier(self, inputs):
        '''
        Classifier defines the classification output.
        This has a set of fully connected layers and a softmax layer.
        '''

        classification_output = tf.keras.layers.Dense(units=self.classes, activation='softmax', name='classification')(inputs)
        return classification_output

    def bounding_box_regression(self, inputs):
        '''
        This function defines the regression output for bounding box prediction. 
        Note that we have four outputs corresponding to (xmin, ymin, xmax, ymax)
        '''

        bounding_box_regression_output = tf.keras.layers.Dense(units=self.bounding_box_shape, activation='linear', name='bounding_box')(inputs)
        return bounding_box_regression_output

    def model(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.dense_layers(x)
        
        classification_output = self.classifier(x)
        bounding_box_output = self.bounding_box_regression(x)

        return tf.keras.models.Model(inputs= inputs, outputs= [classification_output, bounding_box_output])


if __name__ == '__main__':
    batch_size = 64
    mnist_object_detection_dataset = MnistObjectDetectionDataset(batch_size=batch_size)

    # Datasets 
    training_dataset = mnist_object_detection_dataset.get_training_dataset()
    validation_dataset = mnist_object_detection_dataset.get_validation_dataset()

    (training_digits, training_labels, training_bboxes,
    validation_digits, validation_labels, validation_bboxes) = mnist_object_detection_dataset.dataset_to_numpy_util(training_dataset, validation_dataset, 10)

    inputs = tf.keras.layers.Input(shape=(75, 75, 1))
    model = ObjectLocalizationModel(classes=10, bounding_box_shape=4).model(inputs)
    print(model.summary())

    model.compile(
        optimizer='adam',
        loss={
            'classification': 'categorical_crossentropy',
            'bounding_box': 'mse'
        },
        metrics={
            'classification': 'accuracy',
            'bounding_box': 'mse'
        }
    )

    steps_per_epoch = 60000//batch_size  # 60,000 items in this dataset
    validation_steps = 1
    model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=10, validation_data=validation_dataset, validation_steps=validation_steps) 
    loss, classification_loss, bounding_box_loss, classification_accuracy, bounding_box_mse = model.evaluate(validation_dataset, steps=1)
    print("Validation accuracy: ", classification_accuracy)
    print("Validation bounding box mse: ", bounding_box_mse)


    predictions = model.predict(validation_digits, batch_size=64)
    predicted_labels = np.argmax(predictions[0], axis=1)
    predicted_bboxes = predictions[1]

    iou = IntersectionOverUnion().intersection_over_union(predicted_bboxes, validation_bboxes)

    iou_threshold = 0.6

    print("Number of predictions where iou > threshold(%s): %s" % (iou_threshold, (iou >= iou_threshold).sum()))
    print("Number of predictions where iou < threshold(%s): %s" % (iou_threshold, (iou < iou_threshold).sum()))
