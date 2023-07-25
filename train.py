import itertools
import pathlib
import os.path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

DATASET_DIR = pathlib.Path(
    tf.keras.utils.get_file(
        fname="American_Sign_Language_Letters_Multiclass.tar",
        origin="file:./datasets/American_Sign_Language_Letters_Multiclass.tar.gz",
        file_hash="f76def78d7efbfd23ca9340a58fce1026dca21500efa2764caa064fa843fdf23",
        extract=True,
    )
).with_suffix("")

CHECKPOINT_PATH: str = "./checkpoint/"
TFLITE_FNAME: str = "model.tflite"

BATCH_SIZE: int = 64
IMAGE_SIZE: tuple[int, int] = (160, 160)
IMAGE_SHAPE: tuple[int, int, int] = IMAGE_SIZE + (3,)

VALIDATION_SPLIT: float = 0.2
DATA_AUGMENTATION_FACTOR: float = 0.1
DROPOUT_RATE: float = 0.2
L2_REGULARIZATION: float = 0.0001

BASE_LEARNING_RATE: float = 0.005
BASE_LR_DECAY_STEPS: int = 300
BASE_LR_DECAY_RATE: float = 0.85
INITIAL_EPOCHS: int = 64

FINE_TUNE_LEARNING_RATE: float = 0.00005
FINE_TUNE_LR_DECAY_STEPS: int = 200
FINE_TUNE_LR_DECAY_RATE: float = 0.95
FINE_TUNE_EPOCHS: int = 32
FINE_TUNE_AT: int = 80

EARLYSTOP_MIN_DELTA: float = 0.00001
EARLYSTOP_PATIENCE: int = 3

OPTIMIZE_TFLITE: bool = False
NUM_CALIBRATION_EXAMPLES: int = 150
NUM_EVAL_EXAMPLES = 50


def build_dataset(validation_split: float, subset: str) -> tf.data.Dataset:
    return tf.keras.preprocessing.image_dataset_from_directory(  # type: ignore
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
    validation_dataset = validation_dataset.cache().prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    return train_dataset, validation_dataset, class_names


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    base_model: tf.keras.Model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    data_augmentation: tf.keras.Sequential = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(mode="horizontal", input_shape=IMAGE_SHAPE),
            tf.keras.layers.RandomRotation(factor=DATA_AUGMENTATION_FACTOR),
            tf.keras.layers.RandomTranslation(
                height_factor=DATA_AUGMENTATION_FACTOR,
                width_factor=DATA_AUGMENTATION_FACTOR,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=DATA_AUGMENTATION_FACTOR,
                width_factor=DATA_AUGMENTATION_FACTOR,
            ),
        ]
    )

    inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2=L2_REGULARIZATION),
        name="outputs",
    )(x)
    model: tf.keras.Model = tf.keras.Model(inputs, outputs)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        BASE_LEARNING_RATE,
        decay_steps=BASE_LR_DECAY_STEPS,
        decay_rate=BASE_LR_DECAY_RATE,
        staircase=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=lr_schedule),  # type: ignore
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    return base_model, model


def fine_tune_model(base_model: tf.keras.Model, model: tf.keras.Model):
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        FINE_TUNE_LEARNING_RATE,
        decay_steps=FINE_TUNE_LR_DECAY_STEPS,
        decay_rate=FINE_TUNE_LR_DECAY_RATE,
        staircase=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=lr_schedule),  # type: ignore
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
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
    plt.ylim([0.0, 1.0])
    plt.plot(
        [INITIAL_EPOCHS - 1, INITIAL_EPOCHS - 1], plt.ylim(), label="Start Fine Tuning"
    )
    plt.legend(loc="lower left")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.ylim([0.0, 4.0])
    plt.plot(
        [INITIAL_EPOCHS - 1, INITIAL_EPOCHS - 1], plt.ylim(), label="Start Fine Tuning"
    )
    plt.legend(loc="lower left")
    plt.title("Training and Validation Loss")

    plt.xlabel("epoch")
    plt.show()


def get_representative_dataset(train_dataset):
    return itertools.islice(
        ([image[None, ...]] for batch, _ in train_dataset for image in batch),
        NUM_CALIBRATION_EXAMPLES,
    )


def save_model(model):
    tf.saved_model.save(model, CHECKPOINT_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if OPTIMIZE_TFLITE:
        converter.optimizations = set([tf.lite.Optimize.DEFAULT])
        if NUM_CALIBRATION_EXAMPLES:
            converter.representative_dataset = (  # type: ignore
                get_representative_dataset
            )
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

    print(f"Class names:\n{class_names}")

    base_model, model = build_model(len(class_names))
    print(f"Base model layer count: {len(base_model.layers)}")
    model.summary()
    print(f"Trainable variables in our model: {len(model.trainable_variables)}")

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        min_delta=EARLYSTOP_MIN_DELTA,  # type: ignore
        patience=EARLYSTOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        train_dataset,
        callbacks=[earlystop_callback],
        epochs=INITIAL_EPOCHS,
        validation_data=validation_dataset,
    )

    if earlystop_callback.stopped_epoch and earlystop_callback.stopped_epoch > 0:
        if earlystop_callback.best_epoch and earlystop_callback.best_epoch > 0:
            INITIAL_EPOCHS = earlystop_callback.best_epoch + 1
        else:
            INITIAL_EPOCHS = earlystop_callback.stopped_epoch + 1

    base_model, model = fine_tune_model(base_model, model)
    model.summary()
    print(f"Number of trainable variables: {len(model.trainable_variables)}")

    fine_tune_history = model.fit(
        train_dataset,
        callbacks=[earlystop_callback],
        epochs=(INITIAL_EPOCHS + FINE_TUNE_EPOCHS),
        initial_epoch=INITIAL_EPOCHS,
        validation_data=validation_dataset,
    )

    acc: tuple[float] = (
        history.history["accuracy"] + fine_tune_history.history["accuracy"]
    )
    val_acc: tuple[float] = (
        history.history["val_accuracy"] + fine_tune_history.history["val_accuracy"]
    )
    loss: tuple[float] = history.history["loss"] + fine_tune_history.history["loss"]
    val_loss: tuple[float] = (
        history.history["val_loss"] + fine_tune_history.history["val_loss"]
    )

    save_model(model)

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
        probs_tf = model(image[None, ...]).numpy()[0]  # type: ignore
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
