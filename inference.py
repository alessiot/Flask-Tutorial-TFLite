import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os


def read_image(image):
    
    return np.array(Image.open(image))#mpimg.imread(image)


#def format_image(image):
#    return tf.image.resize(image[tf.newaxis, ...], [28, 28]) / 255.0
def format_image(image):
    # Cast image to float32
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  
    # Normalize the image in the range [0, 1]
    image = image/255
    return image

def get_category(img):
    """Write a Function to Predict the Class Name

    Args:
        img [jpg]: image file

    Returns:
        [str]: Prediction
    """

    path = 'static/model/'
    tflite_model_file = 'model.tflite'

    # Load TFLite model and allocate tensors.
    with open(path + tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()

    # Interpreter interface for TensorFlow Lite Models.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Gets model input and output details.
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    input_img = read_image(img)
    format_img = format_image(input_img)
    # Sets the value of the input tensor
    format_img = np.expand_dims(format_img, axis=2)
    format_img = np.expand_dims(format_img, axis=0)
    #print(format_img.shape)
    interpreter.set_tensor(input_index, format_img)
    # Invoke the interpreter.
    interpreter.invoke()

    predictions_array = interpreter.get_tensor(output_index)
    predicted_label = np.argmax(predictions_array)

    with open('static/labels.txt', 'r') as f:    
        class_names = [i.strip() for i in f.readlines()]
        
    return class_names[predicted_label]


def plot_category(img, current_time):
    """Plot the input image

    Args:
        img [jpg]: image file
    """
    read_img = mpimg.imread(img)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(ROOT_DIR + f'/static/images/output_{current_time}.png')
    print(file_path)

    if os.path.exists(file_path):
        os.remove(file_path)

    plt.imsave(file_path, read_img)
