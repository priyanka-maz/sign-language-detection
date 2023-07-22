import os.path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
DATASET_DIR = tf.keras.utils.get_file(
    "flower_photos",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    untar=True,
)
"""
DATASET_DIR = "/home/pgan/.keras/datasets/American_Sign_Language_Letters_Multiclass"

CHECKPOINT_PATH: str = "./checkpoint/"
TFLITE_FNAME: str = "model.tflite"

BATCH_SIZE: int = 32
IMAGE_SIZE: tuple[int, int] = (160, 160)
IMAGE_SHAPE: tuple[int, int, int] = IMAGE_SIZE + (3,)

VALIDATION_SPLIT: float = 0.2
BASE_LEARNING_RATE: float = 6.0e-4
FINE_TUNE_LEARNING_RATE: float = 6.0e-6
INITIAL_EPOCHS: int = 60
FINE_TUNE_EPOCHS: int = 80
FINE_TUNE_AT: int = 100

NUM_EVAL_EXAMPLES = 50


def build_dataset(validation_split: float, subset: str) -> tf.data.Dataset:
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory=DATASET_DIR,
        validation_split=validation_split,
        subset=subset,
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )


def split_dataset(
    validation_split: float,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tuple[str]]:
    train_dataset: tf.data.Dataset = build_dataset(validation_split, "training")
    validation_dataset: tf.data.Dataset = build_dataset(validation_split, "validation")

    class_names: tuple[str] = train_dataset.class_names

    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, class_names


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    base_model: tf.keras.Model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SHAPE, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    data_augmentation: tf.keras.Sequential = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(mode="horizontal", input_shape=IMAGE_SHAPE),
            tf.keras.layers.RandomRotation(factor=0.1),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        ]
    )

    inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001),
        name="outputs",
    )(x)
    model: tf.keras.Model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return base_model, model


def finetune_model(base_model: tf.keras.Model, model: tf.keras.Model):
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        #optimizer=tf.keras.optimizers.RMSprop(learning_rate=(FINE_TUNE_LEARNING_RATE)),
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return base_model, model


def plot_summary(
    acc: tuple[float],
    val_acc: tuple[float],
    loss: tuple[float],
    val_loss: tuple[float],
) -> None:
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.ylim([0.8, 1])
    plt.plot(
        [INITIAL_EPOCHS - 1, INITIAL_EPOCHS - 1], plt.ylim(), label="Start Fine Tuning"
    )
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.ylim([0, 1.0])
    plt.plot(
        [INITIAL_EPOCHS - 1, INITIAL_EPOCHS - 1], plt.ylim(), label="Start Fine Tuning"
    )
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    plt.xlabel("epoch")
    plt.show(block=False)


def save_model():
    tf.saved_model.save(model, CHECKPOINT_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter.optimizations = set([tf.lite.Optimize.DEFAULT])
    lite_model_content = converter.convert()

    with open(os.path.join(CHECKPOINT_PATH, TFLITE_FNAME), "wb") as f:
        f.write(lite_model_content)


def load_model():
    interpreter = tf.lite.Interpreter(
        model_path=os.path.join(CHECKPOINT_PATH, TFLITE_FNAME)
    )
    return interpreter


def lite_model(interpreter, images):
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], images)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])


if __name__ == "__main__":
    train_dataset, validation_dataset, class_names = split_dataset(VALIDATION_SPLIT)

    base_model, model = build_model(len(class_names))
    model.summary()
    print(f"Trainable variables in our model: {len(model.trainable_variables)}")

    history = model.fit(
        train_dataset,
        epochs=INITIAL_EPOCHS,
        validation_data=validation_dataset
    )

    base_model, model = finetune_model(base_model, model)
    model.summary()
    print(f"Number of trainable variables: {len(model.trainable_variables)}")

    fine_tune_history = model.fit(
        train_dataset,
        epochs=(INITIAL_EPOCHS + FINE_TUNE_EPOCHS),
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset,
    )

    acc: tuple[float] = (
        history.history["accuracy"] + fine_tune_history.history["accuracy"]
    )
    val_acc: tuple[float] = (
        history.history["val_accuracy"] + fine_tune_history.history["val_accuracy"]
    )
    loss: tuple[float] = (
        history.history["loss"] + fine_tune_history.history["loss"]
    )
    val_loss: tuple[float] = (
        history.history["val_loss"] + fine_tune_history.history["val_loss"]
    )

    save_model()

    plot_summary(acc, val_acc, loss, val_loss)

    interpreter = load_model()

    eval_dataset = (
        (image, label) for batch in train_dataset for (image, label) in zip(*batch)
    )
    count = 0
    count_lite_tf_agree = 0
    count_lite_correct = 0
    count_tf_correct = 0
    for image, label in eval_dataset:
        probs_lite = lite_model(interpreter, image[None, ...])[0]
        probs_tf = model(image[None, ...]).numpy()[0]
        y_lite = np.argmax(probs_lite)
        y_tf = np.argmax(probs_tf)
        y_true = np.argmax(label)
        count += 1
        if y_lite == y_tf:
            count_lite_tf_agree += 1
        if y_lite == y_true:
            count_lite_correct += 1
        if y_tf == y_true:
            count_tf_correct += 1
        if count >= NUM_EVAL_EXAMPLES:
            break
    print(
        f"""
        TFLite model agrees with original model on {count_lite_tf_agree}
        of {count} examples ({100.0 * count_lite_tf_agree / count}%).
        """
    )
    print(
        f"""
        TFlow model is accurate on {count_tf_correct}
        of {count} examples ({100.0 * count_tf_correct / count}%)."""
    )
    print(
        f"""
        TFLite model is accurate on {count_lite_correct}
        of {count} examples ({100.0 * count_lite_correct / count}%)."""
    )
