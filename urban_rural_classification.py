
##Rishikesh Kakade
##Saturday Feburary 9, 2019
##
##Binary Image Classification of Urban and
##Rural Aerial Imagery Using Convolutional
##Neural Networks

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.models import model_from_json

def generate_classifier():

    #generate binary classification model, add conv laers, flatten, join layers
    class_model = Sequential()
    class_model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    class_model.add(MaxPooling2D(pool_size = (2, 2)))
    class_model.add(Conv2D(32, (3, 3), activation = 'relu'))
    class_model.add(MaxPooling2D(pool_size = (2, 2)))
    class_model.add(Flatten())
    class_model.add(Dense(units = 128, activation = 'relu'))
    class_model.add(Dense(units = 1, activation = 'sigmoid'))
    class_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    from keras.preprocessing.image import ImageDataGenerator
    
    #input training imagery, validate
    gen_train_data = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    gen_test_data = ImageDataGenerator(rescale = 1./255)
    training_set = gen_train_data.flow_from_directory(r"C:\Users\Rishi\Documents\image_ret",
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')
    test_set = gen_test_data.flow_from_directory(r"C:\Users\Rishi\Documents\test", target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')
    class_model.fit_generator(training_set,
    steps_per_epoch = 2000,
    epochs = 13,
    verbose = 2,
    validation_data = test_set,
    validation_steps = 200)

    #serialize model to json
    class_model_json = class_model.to_json()
    with open("class_model_json", "w") as json_file:
        json_file.write(class_model_json)

    #serialize weightings to hdf5
    class_model.save_weights("urban_rural_cnn_model.h5")
    print("Model Saved")


def predict():

    from keras.preprocessing.image import ImageDataGenerator
    
    gen_train_data = ImageDataGenerator(rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
    training_set = gen_train_data.flow_from_directory(r"\train_set",target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')

    # load json and create model
    json_file = open(r"\class_model_json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    # load weights into new model
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("urban_rural_cnn_model.h5")
    print("Loaded model from disk")

    #predict
    from keras.preprocessing import image

    for q in range(0,200):
        test_image = image.load_img(r"\test\test1\val." +str(q) +".jpg", target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        training_set.class_indices

        print(q)
        if result[0][0] == 1:
            prediction = 'rural\n'
        else:
            prediction = 'urban\n'

        #print(prediction)

        return prediction



    
