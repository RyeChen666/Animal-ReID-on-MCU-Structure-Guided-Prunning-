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
import keras_cv  # 仍保留，方便你后续随时启用增强
from mean_average_precision import (
    evaluate_model_open_set,
    MeanAveragePrecisionCallback
)
from pathlib import Path
from absl import logging
import json


# ========== 这里指定要加载的本地 SavedModel 目录 ==========
# 如果就是你之前用这段脚本保存的模型，默认就是这个路径
LOCAL_SAVEDMODEL_DIR = "distill_results/stu_dir"


def get_model_size(path: str) -> float:
    return round(sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    ) / (1024 * 1024), 2)  # MB


# 保留原来的增强函数（当前脚本未启用，方便你以后打开）
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

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(transformations),
                     size=min(augmentation_count, len(transformations)),
                     replace=False)
    for i in idx:
        x = transformations[i](x)
    return x


# 训练容器：保持不变（TripletSemiHardLoss）
class DistillModel(tf.keras.Model):
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

    # dataset_path = "dataset/CornwallCattle"
    dataset_path = "dataset/cw_simple"

    dataset_name = Path(dataset_path).name
    logging.info(f"Starting experiment for {dataset_name}.")

    # ========== 关键修改：加载你本地之前保存的学生模型 ==========
    # 注意：compile=False，避免加载旧的优化器状态
    student = tf.keras.models.load_model(LOCAL_SAVEDMODEL_DIR, compile=False)
    student.summary()
    
    # ========== 冻结 MobileNetV2 backbone，只训练 embedding 头 ==========
    backbone = student.get_layer("model")  # 名字来自你贴的 summary
    backbone.trainable = False

    print("[INFO] 冻结 backbone: mobilenetv2_0.50_224")
    print("[INFO] 仅训练 global_average_pooling2d / dropout / dense / lambda 等头部层。")

    # 可选：检查一下 trainable 参数数量
    print("[INFO] trainable variables:", len(student.trainable_variables))
    print("[INFO] non-trainable variables:", len(student.non_trainable_variables))

    
    try:
        print(f"[INFO] Loaded SavedModel from: {LOCAL_SAVEDMODEL_DIR}")
        print(f"[INFO] Model input shape: {student.input_shape}")
        print(f"[INFO] Model output shape: {student.output_shape}")
    except Exception:
        pass

    # 用模型输入尺寸自动设定 image_size
    inp = student.input_shape
    if isinstance(inp, list):
        inp = inp[0]
    # 形如 (None, H, W, C)
    _, H, W, C = inp
    if (H is None) or (W is None):
        H, W = 64, 64   # 如果模型是动态尺寸，这里退回你原先用的 64x64
        print(f"[WARN] SavedModel 未固定输入尺寸，使用默认 image_size={(H, W)}")
    else:
        print(f"[INFO] 使用模型输入尺寸 image_size={(H, W)}")

    split_path = Path(dataset_path)
    ds_opts = dict(
        image_size=(H, W),
        batch_size=32,
        seed=0,
    )

    # ========== 数据集构建：保持不变 ==========
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

    # tf.data 可复现
    options = tf.data.Options()
    options.experimental_deterministic = True
    train_ds  = train_ds.with_options(options)
    query_ds  = query_ds.with_options(options)
    gallery_ds = gallery_ds.with_options(options)

    time_start = time.perf_counter()
    top_k = (1, 5, 10)

    results_dir = Path("distill_results")
    results_dir.mkdir(exist_ok=True)

    # ========== 训练流程：保持不变 ==========
    model = DistillModel(student)
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

    # ========== 评估流程：保持不变 ==========
    print("open evaluation results...")
    best_student = tf.keras.models.load_model("stu_dir", compile=False)
    with tf.device('/CPU:0'):
        # 下面这行与你原逻辑一致（如果想更严谨，可把 dataset_name 传进去）
        eval_results = evaluate_model_open_set(
            best_student, query_ds, gallery_ds, top_k=top_k,
            dataset_name="Stoat_sorted_by_id"  # 可选：换成 dataset_name
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
