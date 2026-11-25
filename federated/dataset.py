"""
federated/dataset.py

Tiện ích build PyTorch Dataset/DataLoader để huấn luyện cục bộ (mục 4.2-4.4
đề cương). Đây KHÔNG phải một phần công thức toán học, chỉ là lớp keo nối
DataFrame (từ data/loaders.py, data/poison.py, data/partition.py) với
model + tokenizer answerdotai/ModernBERT-base (models/modernbert_lora.py).

Mọi hàm ở đây dùng torch.Generator CỤC BỘ cho việc shuffle (nếu seed được
truyền vào), KHÔNG BAO GIỜ gọi torch.manual_seed() global -- đúng quy ước
random seed của dự án (xem /areas/st-fedlora-progress.md).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


class TextClassificationDataset(Dataset):
    """
    Bọc 1 DataFrame có cột "text" (str) và "label" (int, đã map 0..C-1,
    xem data/loaders.py/data/poison.py) thành Dataset tokenize-on-the-fly.
    """

    def __init__(self, df, tokenizer, max_length: int = 256):
        # reset_index để __getitem__ luôn đánh chỉ số 0..len-1 liên tục,
        # bất kể index gốc của df (df thường là 1 lát cắt .loc[indices] từ
        # partition, nên index KHÔNG liên tục).
        df = df.reset_index(drop=True)
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_dataloader(
    df,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    seed: Optional[int] = None,
    max_length: int = 256,
) -> DataLoader:
    """
    seed: seed CỤC BỘ cho việc xáo trộn minibatch, chỉ có ý nghĩa khi
    shuffle=True (Train/Fit). Validation/Test luôn shuffle=False, seed bị
    bỏ qua (không ảnh hưởng tới kết quả, chỉ ghi để rõ ràng lời gọi).
    """
    dataset = TextClassificationDataset(df, tokenizer, max_length=max_length)
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )
