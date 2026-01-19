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


import os
import time

import numpy as np
import tensorflow as tf
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

alpha_beta_combinations = [
    (1.0, 16.0),
]

def augmentation_layer(
    x: tf.Tensor,
    augmentation_count: int = 4,
    augmentation_factor: float = 0.1,
    seed: int = 0,
):
    if augmentation_count == 0 or augmentation_factor < 1e-6:
        return x

    t_opts = dict(seed=seed)
    transformations = [
        tf.keras.layers.Dropout(augmentation_factor, **t_opts),
        tf.keras.layers.GaussianNoise(augmentation_factor, **t_opts),
        tf.keras.layers.RandomFlip("horizontal", **t_opts),
        tf.keras.layers.RandomTranslation(augmentation_factor, augmentation_factor, fill_mode="constant", **t_opts),
        tf.keras.layers.RandomRotation(augmentation_factor, fill_mode="constant", **t_opts),
        tf.keras.layers.RandomZoom(augmentation_factor, augmentation_factor, fill_mode="constant", **t_opts),
        keras_cv.layers.RandomHue(augmentation_factor, [0, 255], **t_opts),
        keras_cv.layers.RandomSaturation(augmentation_factor, **t_opts),
    ]

    # --- 关键修改：用“局部 RNG”固定选择 ---
    import numpy as np
    rng = np.random.RandomState(seed)  # 只受传入 seed 影响
    idx = rng.choice(len(transformations),
                     size=min(augmentation_count, len(transformations)),
                     replace=False)
    for i in idx:
        x = transformations[i](x)

    return x


# Student Model
def build_student_model(embedding_size=128, input_shape=(64, 64, 3), dropout_rate=0.1, augmentation_count=4, augmentation_factor=0.1, seed=0):
    # inputs = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
    # x = augmentation_layer(inputs, augmentation_count=augmentation_count, augmentation_factor=augmentation_factor, seed=seed)
    # x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    base_model_full = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape, alpha=1.0)
    # base_model_full = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=input_shape, alpha=0.2)
    # base_model_full.summary()

    # # 截断到第 71 层（保留前 72 层）
    # truncated_output = base_model_full.layers[71].output
    # truncated_model = tf.keras.Model(inputs=base_model_full.input, outputs=truncated_output)

    # # 将截断模型应用于输入
    # truncated_model.trainable = True
    # x = truncated_model(x, training=True)
    


    x = base_model_full(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate, seed=seed)(x)
    x = tf.keras.layers.Dense(embedding_size, activation=None)(x)
    x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="student_model")

    # return tf.keras.Model(inputs=inputs, outputs=x, name="student_model")
    # print(f"截断后的模型共 {len(truncated_model.layers)} 层")
    print(f"原始模型共 {len(base_model_full.layers)} 层")
    return model
    # return model, base_model_full


# Distillation Model
class DistillModel(tf.keras.Model):
    def __init__(self, student, teacher, alpha=1.0, beta=0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.triplet_loss_fn = tfa.losses.TripletSemiHardLoss()
        self.distill_loss_fn = tf.keras.losses.MeanSquaredError()
        self.alpha = alpha
        self.beta = beta

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            student_embed = self.student(x, training=True)
            teacher_embed = tf.stop_gradient(self.teacher(x, training=False))
            triplet_loss = self.triplet_loss_fn(y, student_embed)
            distill_loss = self.distill_loss_fn(teacher_embed, student_embed)
            total_loss = self.alpha * triplet_loss + self.beta * distill_loss
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        return {"loss": total_loss, "triplet_loss": triplet_loss, "distill_loss": distill_loss}

    def call(self, inputs):
        return self.student(inputs)


# Compute the model size
def get_model_size(path: str) -> float:
    return round(sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    ) / (1024 * 1024), 2)  # MB


def main():
    logging.set_verbosity(logging.INFO)
    # dataset_path = "dataset/atrw_sorted_by_id"
    # dataset_path = "dataset/mpdd_sorted_by_id"
    # dataset_path = "dataset/friesiancattle2017_sorted_by_id"
    # dataset_path = "dataset/lion_sorted_by_id"
    # dataset_path = "dataset/Stoat_sorted_by_id"
    dataset_path = "dataset/CoBRA"
    # dataset_path = "dataset/IPanda50"

    dataset_name = Path(dataset_path).name
    logging.info(f"Starting experiment for {dataset_name}.")

    split_path = Path(dataset_path)

    ds_opts = dict(
        image_size=(64, 64),
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
    
    # Make tf.data deterministic
    options = tf.data.Options()
    options.experimental_deterministic = True
    train_ds  = train_ds.with_options(options)
    query_ds  = query_ds.with_options(options)
    gallery_ds = gallery_ds.with_options(options)

    time_start = time.perf_counter()

    top_k = (1, 5, 10)

    results_dir = Path("model_64_atrw_model")
    results_dir.mkdir(exist_ok=True)

    for alpha, beta in alpha_beta_combinations:
        logging.info(f"Training with alpha={alpha}, beta={beta}")

        # loading teacher model
        teacher = tf.keras.models.load_model("model_64/cow/model")
        # teacher.summary()

        teacher.trainable = False

        # building the student model
        student = build_student_model()
        model = DistillModel(student, teacher, alpha=alpha, beta=beta)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

        callbacks = [
            MeanAveragePrecisionCallback(query_ds=query_ds, gallery_ds=gallery_ds, top_k=top_k, patience=10, save_path="best_student_model")
        ]
        model.fit(
            train_ds.cache().prefetch(tf.data.AUTOTUNE),
            epochs=50,
            callbacks=callbacks,
            verbose=1,
        )


        time_train = time.perf_counter() - time_start

        model_name = f"student_a{alpha}_b{beta}".replace(".", "_")
        model_path = results_dir / model_name
        model.student.trainable = False
        logging.info(f"Saving student model to {model_path}")
        model.student.save(model_path)

        # Evaluation
        print("open evaluation results...")
        best_student = tf.keras.models.load_model("best_student_model")
        with tf.device('/CPU:0'):
            eval_results = evaluate_model_open_set(best_student, query_ds, gallery_ds, top_k=top_k)
        
        results = {
            "dataset": Path(dataset_path).name,
            "alpha": alpha,
            "beta": beta,
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
