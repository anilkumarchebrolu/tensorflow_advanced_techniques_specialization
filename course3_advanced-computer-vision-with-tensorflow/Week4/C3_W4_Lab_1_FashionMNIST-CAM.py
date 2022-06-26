from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential, Model
import scipy as sp
from PIL import Image
import os


def load_fashion_mnist_and_preprocess():
    ## Download and prepare mnist data
    # load the Fashion MNIST dataset
    (X_train,Y_train),(X_test,Y_test)  = fashion_mnist.load_data()

    # Put an additional axis for the channels of the image.
    # Fashion MNIST is grayscale so we place 1 at the end. Other datasets
    # will need 3 if it's in RGB.
    X_train = X_train.reshape(60000,28,28,1)
    X_test = X_test.reshape(10000,28,28,1)

    # Normalize the pixel values from 0 to 1
    X_train = X_train/255
    X_test  = X_test/255

    # Cast to float
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')
    return ((X_train, Y_train), (X_test, Y_test))


def show_img(img):
    '''utility function for reshaping and displaying an image'''

    # convert to float array if img is not yet preprocessed
    img  = np.array(img,dtype='float')

    # remove channel dimension
    img = img.reshape((28,28))

    # display image
    plt.imshow(img)
    plt.show()

def sample_model():
    # use the Sequential API
    model = Sequential()

    # notice the padding parameter to recover the lost border pixels when doing the convolution
    model.add(Conv2D(16,input_shape=(28,28,1),kernel_size=(3,3),activation='relu',padding='same'))
    # pooling layer with a stride of 2 will reduce the image dimensions by half
    model.add(MaxPooling2D(pool_size=(2,2)))

    # pass through more convolutions with increasing filters
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))

    # use global average pooling to take into account lesser intensity pixels
    model.add(GlobalAveragePooling2D())

    # output class probabilities
    model.add(Dense(10,activation='softmax'))

    model.summary()
    return model


def generating_class_activation_map_CAM(cam_model, feature_idx, features, results):
    # these are the weights going into the softmax layer
    last_dense_layer = model.layers[-1]

    # get the weights list.  index 0 contains the weights, index 1 contains the biases
    gap_weights_l = last_dense_layer.get_weights()

    print("gap_weights_l index 0 contains weights ", gap_weights_l[0].shape)
    print("gap_weights_l index 1 contains biases ", gap_weights_l[1].shape)

    # shows the number of features per class, and the total number of classes
    # Store the weights
    gap_weights = gap_weights_l[0]

    print(f"There are {gap_weights.shape[0]} feature weights and {gap_weights.shape[1]} classes.")

    features_for_img = features[feature_idx,:,:,:]
    print(f"The features for image index {feature_idx} has shape (height, width, num of feature channels) : ", features_for_img.shape)

    features_for_img_scaled = sp.ndimage.zoom(features_for_img, (28/3, 28/3,1), order=2)

    # Select the weights that are used for a specific class (0...9)
    class_id = 0
    # take the dot product between the scaled image features and the weights for 
    gap_weights_for_one_class = gap_weights[:,class_id]

    print("features_for_img_scaled has shape ", features_for_img_scaled.shape)
    print("gap_weights_for_one_class has shape ", gap_weights_for_one_class.shape)
    # take the dot product between the scaled features and the weights for one class
    cam = np.dot(features_for_img_scaled, gap_weights_for_one_class)

    print("class activation map shape ", cam.shape)
    return gap_weights


def save_cam(image_index, gap_weights, features, results):
    '''displays the class activation map of a particular image'''

    # takes the features of the chosen image
    features_for_img = features[image_index,:,:,:]

    # get the class with the highest output probability
    prediction = np.argmax(results[image_index])

    # get the gap weights at the predicted class
    class_activation_weights = gap_weights[:,prediction] # global average precision weights

    # upsample the features to the image's original size (28 x 28)
    class_activation_features = sp.ndimage.zoom(features_for_img, (28/3, 28/3, 1), order=2)

    # compute the intensity of each feature in the CAM
    cam_output  = np.dot(class_activation_features,class_activation_weights)
    
    print('Predicted Class = ' +str(prediction)+ ', Probability = ' + str(results[image_index][prediction]))
    
    # show the upsampled image
    plt.imshow(np.squeeze(X_test[image_index],-1), alpha=0.5)
    
    # strongly classified (95% probability) images will be in green, else red
    if results[image_index][prediction]>0.95:
        cmap_str = 'Greens'
    else:
        cmap_str = 'Reds'

    # overlay the cam output
    plt.imshow(cam_output, cmap=cmap_str, alpha=0.5)

    # display the image
    plt.savefig(os.path.join("outputs", str(image_index)+".jpg"))

def save_maps(desired_class, num_maps, gap_weights, features, results):
    '''
    goes through the first 10,000 test images and generates CAMs 
    for the first `num_maps`(int) of the `desired_class`(int)
    '''

    counter = 0

    if desired_class < 10:
        print("please choose a class less than 10")

    # go through the first 10000 images
    for i in range(0,10000):
        # break if we already displayed the specified number of maps
        if counter == num_maps:
            break

        # images that match the class will be shown
        if np.argmax(results[i]) == desired_class:
            counter += 1
            save_cam(i, gap_weights, features, results)

if __name__ == '__main__':
    ((X_train, Y_train), (X_test, Y_test)) = load_fashion_mnist_and_preprocess()
    show_img(X_train[1])

    # Create a model
    model = sample_model()
    
    # configure the training
    model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

    # train the model. just run a few epochs for this test run. you can adjust later.
    model.fit(X_train, Y_train, batch_size=32, epochs=5, validation_split=0.1, shuffle=True)

    # same as previous model but with an additional output
    
    cam_model  = Model(inputs=model.input,outputs=(model.layers[-3].output,model.layers[-1].output))
    cam_model.summary()


    # get the features and results of the test images using the newly created model
    features, results = cam_model.predict(X_test)

    # shape of the features
    print("features shape: ", features.shape)
    print("results shape", results.shape)

    desired_class = 0
    feature_idx = 0
    num_maps = 20

    gap_weights = generating_class_activation_map_CAM(cam_model, feature_idx, features, results)
    save_maps(desired_class, num_maps, gap_weights, features, results)

