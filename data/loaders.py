"""
data/loaders.py
Đọc và chuẩn hoá 2 tập dữ liệu thực nghiệm (AG News Redis, DBpedia L1) từ
file CSV trên Google Drive và đưa về một schema thống nhất: 2 cột `text`,
`label` (label là số nguyên 0-indexed) để toàn bộ pipeline phía sau
(partition, client, poison, model) không cần quan tâm dataset gốc khác nhau
ra sao.

Quy ước file đầu vào (đặt trong `data_dir`, ví dụ trên Drive):
    agnews_redis/train.csv   agnews_redis/val.csv   agnews_redis/test.csv
    dbpedia/train.csv        dbpedia/val.csv        dbpedia/test.csv

Cả hai dataset đều có 3 phần train/val/test theo mục 4.2 của đề cương:
    - AG News Redis (mục 4.2.1): bản tái phân chia cho nghiên cứu này. Gộp
      train+test gốc của AG News (120.000 + 7.600 = 127.600 mẫu) rồi
      stratified split lại theo tỉ lệ 70/20/10 (xem
      `data/prepare_ag_news_redis.py`). KHÔNG dùng split gốc của AG News
      (vốn chỉ có 2 phần train/test, không có val).
    - DBpedia L1 (mục 4.2.2): giữ nguyên 3 split train/val/test 70/20/10 gốc
      của bộ dữ liệu nguồn.

AG News Redis -- mỗi file 3 cột, không có header (giữ đúng định dạng cột của
AG News gốc vì AG News Redis chỉ tái phân chia theo dòng, không đổi schema):
    Class Index (1-4), Title, Description
    1=World, 2=Sports, 3=Business, 4=Sci/Tech

DBpedia (L1) — 2 cột, không có header:
    text, L1
    L1 ∈ {Agent, Device, Event, Place, Species, SportsSeason,
          TopicalConcept, UnitOfWork, Work}   (9 nhãn)

Lưu ý: nếu file CSV gốc có dòng header thì hàm đọc vẫn tự phát hiện
(xem `_read_csv_auto_header`) nên không cần chỉnh sửa file tay.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

try:
    from datasets import Dataset
except ImportError:  # pragma: no cover - datasets chỉ cần khi tokenize
    Dataset = None


# AG News Redis: id gốc trong file là 1-4 -> trừ 1 để thành 0-3
AGNEWS_LABELS: dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

# DBpedia L1: 9 nhãn, thứ tự alphabet để cố định ánh xạ string -> int
DBPEDIA_L1_LABELS: dict[int, str] = {
    0: "Agent",
    1: "Device",
    2: "Event",
    3: "Place",
    4: "Species",
    5: "SportsSeason",
    6: "TopicalConcept",
    7: "UnitOfWork",
    8: "Work",
}
_DBPEDIA_NAME_TO_ID: dict[str, int] = {v: k for k, v in DBPEDIA_L1_LABELS.items()}

# Khoá "agnews_redis" là tên dataset CHUẨN dùng trong toàn bộ pipeline, theo
# đúng mục 4.2.1 của đề cương (đây không phải split gốc của AG News). Vẫn giữ
# alias "agnews" trỏ tới cùng hàm đọc/label map để tương thích ngược nếu có
# code cũ nào còn gọi dataset_name="agnews".
DATASET_LABELS: dict[str, dict[int, str]] = {
    "agnews_redis": AGNEWS_LABELS,
    "agnews": AGNEWS_LABELS,
    "dbpedia": DBPEDIA_L1_LABELS,
}
DATASET_NUM_CLASSES: dict[str, int] = {
    "agnews_redis": len(AGNEWS_LABELS),
    "agnews": len(AGNEWS_LABELS),
    "dbpedia": len(DBPEDIA_L1_LABELS),
}

# Cả 2 dataset dùng trong nghiên cứu đều có 3 split train/val/test (mục 4.2).
VALID_SPLITS = ("train", "val", "test")


def _read_csv_auto_header(path: str, expected_ncols: int) -> pd.DataFrame:
    """
    Đọc CSV mà không giả định có header. Nếu dòng đầu tiên trông giống header
    (toàn chữ, không phải số ở cột nhãn) thì bỏ dòng đó đi.
    """
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if df.shape[1] != expected_ncols:
        raise ValueError(
            f"[{path}] có {df.shape[1]} cột, nhưng kỳ vọng {expected_ncols} cột."
        )
    first_row_col0 = str(df.iloc[0, 0]).strip()
    looks_like_header = not first_row_col0.lstrip("-").isdigit() and expected_ncols == 3
    # Với dbpedia (2 cột: text, L1) cột 0 luôn là text (chuỗi) nên không dùng
    # cách check trên. Thay vậy thì check xem cột cuối dòng đầu có nằm trong
    # tập nhãn hợp lệ hay là chuỗi "L1"/"label" (dấu hiệu header) không.
    if expected_ncols == 2:
        last_val = str(df.iloc[0, -1]).strip()
        looks_like_header = last_val not in _DBPEDIA_NAME_TO_ID
    if looks_like_header:
        df = df.iloc[1:].reset_index(drop=True)
    return df


def _require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}\n"
            f"Hãy chắc chắn đã tải dữ liệu về và đặt đúng cấu trúc thư mục "
            f"(xem docstring đầu file loaders.py)."
        )


def load_agnews_csv(path: str) -> pd.DataFrame:
    """
    Đọc 1 file AG News Redis (train.csv, val.csv hoặc test.csv -- xem
    `data/prepare_ag_news_redis.py` để sinh 3 file này từ AG News gốc).
    Cột gốc: Class Index (1-4), Title, Description
    -> Title + Description được nối thành `text`; label = Class Index - 1 (0-3)
    """
    _require_file(path)
    raw = _read_csv_auto_header(path, expected_ncols=3)
    raw.columns = ["class_index", "title", "description"]

    class_idx = raw["class_index"].astype(int)
    if class_idx.min() < 1 or class_idx.max() > 4:
        raise ValueError(
            f"[{path}] Class Index phải trong khoảng 1-4, "
            f"nhưng gặp giá trị [{class_idx.min()}, {class_idx.max()}]."
        )

    title = raw["title"].astype(str).str.strip()
    desc = raw["description"].astype(str).str.strip()
    text = (title + ". " + desc).str.strip()

    out = pd.DataFrame({
        "text": text,
        "label": (class_idx - 1).astype(int),
    })
    return out.reset_index(drop=True)


def load_dbpedia_csv(path: str) -> pd.DataFrame:
    """
    Đọc 1 file DBpedia L1 (train.csv, val.csv hoặc test.csv là 3 split gốc
    của bộ dữ liệu, xem mục 4.2.2).
    Cột gốc: text, L1 (chuỗi tên nhãn, 9 giá trị, xem DBPEDIA_L1_LABELS)
    -> label = ánh xạ chuỗi L1 sang id 0-8 theo DBPEDIA_L1_LABELS cố định.
    """
    _require_file(path)
    raw = _read_csv_auto_header(path, expected_ncols=2)
    raw.columns = ["text", "l1"]

    l1 = raw["l1"].astype(str).str.strip()
    unknown = set(l1.unique()) - set(_DBPEDIA_NAME_TO_ID.keys())
    if unknown:
        raise ValueError(
            f"[{path}] Gặp nhãn L1 không nằm trong 9 nhãn đã định nghĩa: {unknown}\n"
            f"Nhãn hợp lệ: {list(_DBPEDIA_NAME_TO_ID.keys())}"
        )

    out = pd.DataFrame({
        "text": raw["text"].astype(str).str.strip(),
        "label": l1.map(_DBPEDIA_NAME_TO_ID).astype(int),
    })
    return out.reset_index(drop=True)


_LOADER_FN = {
    "agnews_redis": load_agnews_csv,
    "agnews": load_agnews_csv,
    "dbpedia": load_dbpedia_csv,
}


def load_dataset_split(dataset_name: str, split: str, data_dir: str) -> pd.DataFrame:
    """
    dataset_name: "agnews_redis" | "dbpedia" (hoặc alias "agnews")
    split: "train" | "val" | "test"
    data_dir: thư mục gốc chứa <dataset_name>/{train,val,test}.csv
              (ví dụ: "/content/drive/MyDrive/st-fedlora-data")
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in _LOADER_FN:
        raise ValueError(f"dataset_name phải là {list(_LOADER_FN)}, nhận '{dataset_name}'")
    if split not in VALID_SPLITS:
        raise ValueError(f"split phải là {VALID_SPLITS}, nhận '{split}'")

    path = os.path.join(data_dir, dataset_name, f"{split}.csv")
    df = _LOADER_FN[dataset_name](path)
    df.attrs["dataset_name"] = dataset_name
    df.attrs["label_names"] = DATASET_LABELS[dataset_name]
    return df


@dataclass
class DatasetBundle:
    """Gói train/val/test DataFrame đã chuẩn hoá cùng với metadata nhãn.

    Cả AG News Redis và DBpedia L1 đều dùng 3 split (mục 4.2):
        - `val_df` ứng với Global Validation: dùng để chọn checkpoint
          Classification Head warm-up (mục 4.3.1), chọn λ_poison/τ/σ/γ_trust
          (mục 5, 7.1), không dùng để tính gradient cục bộ hay tổng hợp
          tham số.
        - `test_df` ứng với Global Test: chỉ dùng để báo cáo ACC/ASR cuối
          cùng (mục 7.1 bước 5); không dùng để chọn checkpoint hay siêu
          tham số.
    """
    name: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    label_names: dict[int, str] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


def load_dataset_bundle(dataset_name: str, data_dir: str) -> DatasetBundle:
    train_df = load_dataset_split(dataset_name, "train", data_dir)
    val_df = load_dataset_split(dataset_name, "val", data_dir)
    test_df = load_dataset_split(dataset_name, "test", data_dir)
    return DatasetBundle(
        name=dataset_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_names=DATASET_LABELS[dataset_name.lower()],
    )


def tokenize_dataframe(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 128,
    text_col: str = "text",
    label_col: str = "label",
) -> "Dataset":
    """
    Chuyển DataFrame (text, label) thành datasets.Dataset đã tokenize sẵn
    (input_ids, attention_mask, labels) để đưa thẳng vào ModernBERT.
    """
    if Dataset is None:
        raise ImportError("Cần cài package `datasets` (pip install datasets).")

    hf_ds = Dataset.from_pandas(df[[text_col, label_col]].reset_index(drop=True))

    def _tokenize_batch(batch):
        enc = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        enc["labels"] = batch[label_col]
        return enc

    hf_ds = hf_ds.map(_tokenize_batch, batched=True, remove_columns=[text_col, label_col])
    hf_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return hf_ds


# Self-test nhanh (chạy: python -m data.loaders --data_dir <path>)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kiểm tra nhanh loaders.py với dữ liệu thật")
    parser.add_argument("--data_dir", required=True, help="Thư mục gốc chứa agnews_redis/ và dbpedia/")
    args = parser.parse_args()

    for name in ("agnews_redis", "dbpedia"):
        print(f"\n=== {name} ===")
        bundle = load_dataset_bundle(name, args.data_dir)
        print(f"train: {len(bundle.train_df)} mẫu | val: {len(bundle.val_df)} mẫu | "
              f"test: {len(bundle.test_df)} mẫu")
        print(f"num_classes: {bundle.num_classes} -> {bundle.label_names}")
        print("Phân bố nhãn (train):")
        print(bundle.train_df["label"].value_counts().sort_index().rename(
            index=lambda i: f"{i} ({bundle.label_names[i]})"
        ))
        print(bundle.train_df.head(2))
