import math
import numpy as np
import tensorflow as tf
import functools, multiprocessing
import os, shutil
from typing import Tuple, Callable

def _hitcount_embedding_open_set(
    i: int,
    y_true: np.ndarray = None,
    y_pred: np.ndarray = None,
    z_true: np.ndarray = None,
    z_pred: np.ndarray = None,
    top_k: tuple[int] = None,
    dataset_name: str | None = None,
) -> dict[int, int]:
  y1 = y_true[i]

  same = np.where(z_true == y1)[0].tolist()
  diff = np.where(z_true != y1)[0].tolist()

  if len(diff) > 0 and len(same) > 0:
    dist_diff = [(np.linalg.norm(y_pred[i] - z_pred[j]), 0, j) for j in diff]
    is_stoat = (dataset_name == "Stoat_sorted_by_id" and y_true.shape[0] == z_true.shape[0])
    if is_stoat:
      # 自匹配 j == i 的距离置为 +inf（等价于“改对角线为最大值/无穷大”）
      dist_same = [
          (np.inf if j == i else np.linalg.norm(y_pred[i] - z_pred[j]), 1, j)
          for j in same
      ]
      print(111)
    else:
      # print(222)
      dist_same = [(np.linalg.norm(y_pred[i] - z_pred[j]), 1, j) for j in same]
    # dist_same = [(np.linalg.norm(y_pred[i] - z_pred[j]), 1, j) for j in same]
    dist_combined = dist_diff + dist_same

    dist_combined.sort(key=lambda t: (t[0], t[2]))
    hits_sorted = [h for _, h, _ in dist_combined]

    # CMC@K
    cmc = {k: 1 if sum(hits_sorted[:k]) > 0 else 0 for k in top_k}

    # 严格版 AP（分母=len(same)）
    precisions, correct = [], 0
    for rank, h in enumerate(hits_sorted, start=1):
      if h == 1:
        correct += 1
        precisions.append(correct / rank)
    ap = (sum(precisions) / len(same)) if len(same) > 0 else 0.0

    return {**cmc, "AP": ap}
  else:
    print("1111")
    return {**{k: 0 for k in top_k}, "AP": 0.0}


def _score_embeddings_parallel(score_func, map_iter, **kwargs) -> dict[str, float]:
  top_k = kwargs.pop("top_k")
  
  total = 0
  hits = {k: 0 for k in top_k}
  ap_total = 0.0

  map_func = functools.partial(score_func, **kwargs, top_k=top_k)
  with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    for res in pool.imap_unordered(map_func, map_iter, chunksize=64):
      total += 1
      ap_total += res["AP"]
      for k in top_k:
        hits[k] += res[k]

  result = {f"CMC@{k}": hits[k] / total for k in top_k}
  result["mAP"] = ap_total / total
  return result


def evaluate_model_open_set(
    model: tf.keras.Model,
    query_data: tf.data.Dataset,
    gallery_data: tf.data.Dataset,
    top_k: tuple[int] = (1,),
    dataset_name: str | None = None,
) -> dict[str, float]:
  images_query, labels_query = tuple(zip(*query_data))
  images_gallery, labels_gallery = tuple(zip(*gallery_data))

  # 标签 → numpy 一维
  y_true = tf.concat(labels_query, axis=0).numpy().reshape(-1)
  z_true = tf.concat(labels_gallery, axis=0).numpy().reshape(-1)

  # 一次性前向
  y_pred = model(tf.concat(images_query, axis=0), training=False).numpy()
  z_pred = model(tf.concat(images_gallery, axis=0), training=False).numpy()

  assert y_true.shape[0] == y_pred.shape[0], f"{len(y_true)} != {len(y_pred)}"
  return _score_embeddings_parallel(
      _hitcount_embedding_open_set,
      range(y_true.shape[0]),
      y_true=y_true,
      y_pred=y_pred,
      z_true=z_true,
      z_pred=z_pred,
      top_k=top_k,
      dataset_name=dataset_name
  )


class MeanAveragePrecisionCallback(tf.keras.callbacks.Callback):
    def __init__(self, query_ds, gallery_ds, top_k=(1, 5, 10), patience=5, save_path="best_student_model"):
        super().__init__()
        self.query_ds = query_ds
        self.gallery_ds = gallery_ds
        self.top_k = top_k
        self.patience = patience
        self.save_path = save_path  # 保存最佳模型的路径

        self.best_map = -1.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # 计算 mAP
        with tf.device('/CPU:0'):
            results = evaluate_model_open_set(
                self.model.student, self.query_ds, self.gallery_ds, top_k=self.top_k, dataset_name="Stoat_sorted_by_id"
            )

        current_map = results.get("mAP", 0.0)
        print(f"Epoch {epoch + 1}: mAP = {current_map:.4f}")

        # EarlyStopping + 保存最佳模型
        if current_map > self.best_map:
            self.best_map = current_map
            self.wait = 0
            print(f" New best mAP: {self.best_map:.4f} — saving model to {self.save_path}")
            # 保存完整模型（包含权重 + BN 统计值 + 配置）
            self.model.student.save(self.save_path, include_optimizer=False)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(
                    f" Early stopping at epoch {epoch + 1}, restoring best model with mAP = {self.best_map:.4f}"
                )
                # 恢复最佳模型（重新加载）
                self.model.student = tf.keras.models.load_model(self.save_path)
                self.model.stop_training = True

        logs["mAP"] = current_map
        

# class MeanAveragePrecisionCallback(tf.keras.callbacks.Callback):
#     """
#     每个 epoch 结束：
#       1) 计算 open-set mAP
#       2) 若更好，删除旧目录并以 SavedModel 形式保存（会覆盖）
#       3) 基于 patience 触发早停（不在回调里恢复模型）
#     - query_ds / gallery_ds: tf.data.Dataset，元素为 (images, labels)
#     - model_selector: 选择需要评估/保存的模型（单模型：lambda m: m；蒸馏：lambda m: m.student）
#     - save_dir: 保存 SavedModel 的目录路径（例如 ".../best_by_map_savedmodel"）
#     """
#     def __init__(
#         self,
#         query_ds: tf.data.Dataset,
#         gallery_ds: tf.data.Dataset,
#         top_k: Tuple[int, ...] = (1, 5, 10),
#         patience: int = 5,
#         save_dir: str = "best_by_map_savedmodel",
#         model_selector: Callable[[tf.keras.Model], tf.keras.Model] = lambda m: m,
#         eval_device: str = "/CPU:0",
#         verbose: bool = True,
#     ):
#         super().__init__()
#         self.query_ds = query_ds
#         self.gallery_ds = gallery_ds
#         self.top_k = top_k
#         self.patience = patience
#         self.save_dir = save_dir
#         self.model_selector = model_selector
#         self.eval_device = eval_device
#         self.verbose = verbose

#         self.best_map = -1.0
#         self.wait = 0

#     def _clear_dir(self, path: str):
#         # 跨文件系统更稳妥的删除
#         try:
#             if tf.io.gfile.exists(path):
#                 tf.io.gfile.rmtree(path)
#         except Exception:
#             shutil.rmtree(path, ignore_errors=True)

#     def _save_savedmodel(self, model: tf.keras.Model, path: str):
#         self._clear_dir(path)
#         # tf.keras 环境下，直接 model.save(path) 即 SavedModel（目录）
#         # 若你在 Keras3 环境，可改用：model.export(path)
#         try:
#             model.save(path)   # SavedModel 目录
#         except Exception:
#             # 兼容 Keras3
#             if hasattr(model, "export"):
#                 model.export(path)
#             else:
#                 raise

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         eval_model = self.model_selector(self.model)

#         # 计算 mAP（一般放 CPU，避免与训练抢 GPU）
#         with tf.device(self.eval_device):
#             results = evaluate_model_open_set(
#                 eval_model, self.query_ds, self.gallery_ds, top_k=self.top_k, dataset_name="Stoat_sorted_by_id"
#             )
#         current_map = float(results.get("mAP", 0.0))

#         if self.verbose:
#             print(f"[mAP-ES] Epoch {epoch + 1}: mAP = {current_map:.4f} (best = {self.best_map:.4f})")

#         # 保存最佳（SavedModel 目录；若已存在则先删再存）
#         if current_map > self.best_map:
#             self.best_map = current_map
#             self.wait = 0
#             if self.verbose:
#                 print(f"[mAP-ES] New best mAP — saving SavedModel to {self.save_dir}")
#             self._save_savedmodel(eval_model, self.save_dir)
#         else:
#             self.wait += 1
#             if self.wait >= self.patience:
#                 if self.verbose:
#                     print(f"[mAP-ES] Early stopping at epoch {epoch + 1}. "
#                           f"Best mAP = {self.best_map:.4f}.")
#                 # 不在回调内部恢复；训练结束后由主流程从磁盘加载最佳 SavedModel
#                 self.model.stop_training = True

#         logs["mAP"] = current_map


