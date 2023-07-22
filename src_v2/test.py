import os.path
import numpy as np
import tensorflow as tf

CHECKPOINT_PATH: str = "./checkpoint/"
TFLITE_FNAME: str = "model.tflite"

IMAGE_SIZE: tuple[int, int] = (160, 160)
CLASS_NAMES: list[str] = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


def load_model():
    interpreter = tf.lite.Interpreter(
        model_path=os.path.join(CHECKPOINT_PATH, TFLITE_FNAME)
    )
    return interpreter

def get_image_array():
    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

    img = tf.keras.utils.load_img(
        sunflower_path, target_size=IMAGE_SIZE
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    return img_array


if __name__ == '__main__':
    interpreter = load_model()
    #print(interpreter.get_signature_list())
    classify_lite = interpreter.get_signature_runner('serving_default')

    img_array = get_image_array()
    predictions_lite = classify_lite(input_2=img_array)['outputs']
    score_lite = tf.nn.softmax(predictions_lite)

    print(
        f"""
        This image most likely belongs to {CLASS_NAMES[np.argmax(score_lite)]}
        with a {100 * np.max(score_lite):.2f} percent confidence.
        """
    )
