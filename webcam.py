import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

TFLITE_PATH: str = "./models/model_loss-0.496_accuracy-0.871.tflite"

IMAGE_SIZE: tuple[int, int] = (200, 200)
CLASS_NAMES: list[str] = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y",
    "Z", "del", "nothing", "space",
]

TARGET_FRAME_COUNT: int = 4
TARGET_CONSECUTIVE_PREDICTIONS: int = 3
NOOP_CLASS_NAMES: list[str] = ["nothing"]


def load_model():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    return interpreter


def get_image_array(image_data):
    img_array = tf.keras.utils.img_to_array(image_data)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    return img_array


def predict(image_array):
    predictions_lite = classify_lite(input_2=image_array)["outputs"]
    score_lite = tf.nn.softmax(predictions_lite)

    predicted_char = CLASS_NAMES[np.argmax(score_lite)]
    prediction_score = np.max(score_lite)

    return predicted_char, prediction_score


if __name__ == "__main__":
    interpreter = load_model()
    # print(interpreter.get_signature_list())
    classify_lite = interpreter.get_signature_runner("serving_default")

    x1, y1 = 100, 100
    x2, y2 = (x1 + IMAGE_SIZE[0]), (y1 + IMAGE_SIZE[1])

    video_capture = cv2.VideoCapture(0)

    frame_count: int = 0
    previous_predicted_char: str = ""
    consecutive_predictions: int = 0
    text: str = ""

    while True:
        ret, img = video_capture.read()
        img = cv2.flip(img, 1)

        predicted_char: str = ""
        prediction_score: float = 0.0

        if ret:
            frame_count += 1

            if frame_count == TARGET_FRAME_COUNT:
                frame_count = 0

                img_cropped = img[y1:y2, x1:x2]
                image_data = Image.fromarray(img_cropped)
                image_array = get_image_array(image_data)

                predicted_char, prediction_score = predict(image_array)

                if previous_predicted_char == predicted_char:
                    consecutive_predictions += 1
                else:
                    consecutive_predictions = 0

                previous_predicted_char = predicted_char

                if (
                    consecutive_predictions == TARGET_CONSECUTIVE_PREDICTIONS
                    and predicted_char not in NOOP_CLASS_NAMES
                ):
                    consecutive_predictions = 0

                    if predicted_char == "space":
                        text += " "
                    elif predicted_char == "del":
                        text = text[:-1]
                    else:
                        text += predicted_char

            cv2.putText(
                img,
                predicted_char.upper(),
                (100, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (255, 255, 255),
                4,
            )
            cv2.putText(
                img,
                f"(score = {prediction_score:.2f})",
                (100, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("img", img)

            blank_img = np.zeros((200, 1200, 3), np.uint8)
            cv2.putText(
                blank_img,
                text.upper(),
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imshow("sequence", blank_img)

            keyboard_key = cv2.waitKey(1)
            if keyboard_key == 27:  # when `esc` is pressed
                break

    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()
