import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

def predict_class(path):
    # Load the image
    img = cv2.imread(path)

    # Convert image to RGB
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to the model's input size
    RGBImg = cv2.resize(RGBImg, (224, 224))



    # Normalize the image
    image = np.array(RGBImg) / 255.0

    # Load the model
    new_model = tf.keras.models.load_model("model_binary.h5")

    # Make a prediction
    prediction = new_model.predict(np.array([image]))

    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)


    # Display the image
    plt.imshow(RGBImg)
    # Display the predicted class
    if predicted_class == 1:
        print('Predicted Class: No DR')
        plt.title('Predicted Class: No DR')
    else:
        print('Predicted Class: DR')
        plt.title('Predicted Class: DR')
    plt.show()

# Example usage
predict_class('C:/Users/Bhaskar Marapelli/Downloads/gaussian_filtered_images/gaussian_filtered_images/Mild/0024cdab0c1e.png')
predict_class('C:/Users/Bhaskar Marapelli/Downloads/gaussian_filtered_images/gaussian_filtered_images/No_DR/2ef10194e80d.png')
