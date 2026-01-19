import os
SEED = 0

# 1) env vars BEFORE importing tensorflow
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# 2) python / numpy seeds
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)

# 3) now import tensorflow and set tf/keras seed
import tensorflow as tf
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


import time
import tensorflow_addons as tfa
import dataset
import keras_cv
from mean_average_precision import (
    evaluate_model_open_set,
    MeanAveragePrecisionCallback
)
from pathlib import Path
from absl import logging
import json



LOCAL_SAVEDMODEL_DIR = "results/stu_dir"


def get_model_size(path: str) -> float:
    return round(sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    ) / (1024 * 1024), 2)  # MB

class TrainModel(tf.keras.Model):
    def __init__(self, student):
        super().__init__()
        self.student = student
        self.triplet_loss_fn = tfa.losses.TripletSemiHardLoss()

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            student_embed = self.student(x, training=True)
            triplet_loss = self.triplet_loss_fn(y, student_embed)
            total_loss = triplet_loss
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        return {"loss": total_loss}

    def call(self, inputs):
        return self.student(inputs)


def main():
    logging.set_verbosity(logging.INFO)

    dataset_path = "dataset/CornwallCattle"

    dataset_name = Path(dataset_path).name
    logging.info(f"Starting experiment for {dataset_name}.")

    student = tf.keras.models.load_model(LOCAL_SAVEDMODEL_DIR, compile=False)
    student.summary()
    
    # Freeze the MobileNetV2 backbone and train only the embedding head
    backbone = student.get_layer("model")
    backbone.trainable = False
    print("[INFO] trainable variables:", len(student.trainable_variables))
    print("[INFO] non-trainable variables:", len(student.non_trainable_variables))

    
    try:
        print(f"[INFO] Loaded SavedModel from: {LOCAL_SAVEDMODEL_DIR}")
        print(f"[INFO] Model input shape: {student.input_shape}")
        print(f"[INFO] Model output shape: {student.output_shape}")
    except Exception:
        pass

    inp = student.input_shape
    if isinstance(inp, list):
        inp = inp[0]
    _, H, W, C = inp
    if (H is None) or (W is None):
        H, W = 64, 64
        print(f"[WARN] The SavedModel does not have a fixed input size and uses the default={(H, W)}")
    else:
        print(f"[INFO] Use the model input size image_size={(H, W)}")

    split_path = Path(dataset_path)
    ds_opts = dict(
        image_size=(H, W),
        batch_size=32,
        seed=0,
    )

    train_ds = dataset.triplet_safe_image_dataset_from_directory(
        split_path / "train",
        **ds_opts,
    )
    query_ds = tf.keras.utils.image_dataset_from_directory(
        split_path / "query",
        **ds_opts,
    )
    gallery_ds = tf.keras.utils.image_dataset_from_directory(
        split_path / "gallery",
        **ds_opts,
    )

    options = tf.data.Options()
    options.experimental_deterministic = True
    train_ds  = train_ds.with_options(options)
    query_ds  = query_ds.with_options(options)
    gallery_ds = gallery_ds.with_options(options)

    time_start = time.perf_counter()
    top_k = (1, 5, 10)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    model = TrainModel(student)
    model.student.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    callbacks = [
        MeanAveragePrecisionCallback(
            query_ds=query_ds,
            gallery_ds=gallery_ds,
            top_k=top_k,
            patience=10,
            save_path="stu_dir"
        )
    ]
    model.fit(
        train_ds.cache().prefetch(tf.data.AUTOTUNE),
        epochs=50,
        callbacks=callbacks,
        verbose=1,
    )

    time_train = time.perf_counter() - time_start

    model_name = f"stu_dir"
    model_path = results_dir / model_name
    model.student.trainable = False
    logging.info(f"Saving student model to {model_path}")
    model.student.save(model_path)

    print("open evaluation results...")
    best_student = tf.keras.models.load_model("stu_dir", compile=False)
    with tf.device('/CPU:0'):
        eval_results = evaluate_model_open_set(
            best_student, query_ds, gallery_ds, top_k=top_k,
            dataset_name="CornwallCattle"
        )

    results = {
        "dataset": Path(dataset_path).name,
        "epochs": 50,
        "model_size_mb": get_model_size(str(model_path)),
        "time_train": time_train,
        **eval_results
    }

    with open(results_dir / f"{model_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    logging.info("Experiment %s results: %r", dataset_name, results)


if __name__ == "__main__":
    main()
