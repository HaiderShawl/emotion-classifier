import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('models/saved_model_cnn/my_model')


class emotionClassifier:
    def __init__(self):
        pass
    def predict(self, img):

        class_names = ['disgust', 'happy', 'neutral', 'sad', 'surprise']
        img_path = 'images/validation/surprise/10246.jpg'

        # img = tf.keras.preprocessing.image.load_img(
        #     img_path, target_size=(180, 180)
        # )

        img_array = tf.keras.preprocessing.image.img_to_array(img)

        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        if np.max(score) > 0.8:
            # print(
            #     "This image most likely belongs to {} with a {:.2f} percent confidence."
            #     .format(class_names[np.argmax(score)], 100 * np.max(score))
            # )
            print(class_names[np.argmax(score)])
        else: 
            print('neutral')
