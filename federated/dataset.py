"""
federated/dataset.py

Tiện ích build PyTorch Dataset/DataLoader để huấn luyện cục bộ (mục 4.2-4.4
đề cương). Đây KHÔNG phải một phần công thức toán học, chỉ là lớp keo nối
DataFrame (từ data/loaders.py, data/poison.py, data/partition.py) với
model + tokenizer answerdotai/ModernBERT-base (models/modernbert_lora.py).

Mọi hàm ở đây dùng torch.Generator CỤC BỘ cho việc shuffle (nếu seed được
truyền vào), KHÔNG BAO GIỜ gọi torch.manual_seed() global -- đúng quy ước
random seed của dự án (xem /areas/st-fedlora-progress.md).

CẬP NHẬT (debug hiệu năng, 2026-11-27):
    Bản trước gọi `self.tokenizer(...)` RIÊNG cho từng sample bên trong
    `__getitem__`. Với vài chục nghìn mẫu/client và `padding="max_length"`,
    đây là chi phí CPU rất lớn vì tokenizer (HuggingFace, kể cả bản
    "fast") tốn overhead cố định đáng kể mỗi lần gọi, nhân với số lần gọi
    bằng đúng (số mẫu x số epoch x số round) -- đây là nguyên nhân chính
    khiến chương trình trông như bị "đứng" (không hề deadlock, chỉ là chạy
    cực chậm và không có log nào ở giữa).

    Sửa: tokenize MỘT LẦN DUY NHẤT, theo BATCH, ngay trong __init__. Sau đó
    __getitem__ chỉ cần index vào tensor đã có sẵn (gần như tức thời).
    Đánh đổi: tốn thêm RAM để giữ input_ids/attention_mask đã tokenize cho
    cả DataFrame trong bộ nhớ -- với kích thước dữ liệu của đề tài (vài
    chục nghìn mẫu, max_length <= 256) việc này không đáng kể (vài chục MB
    mỗi Client), nên không cần tokenize lười (lazy) nữa.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm luôn nên có trong môi trường Colab
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        """Fallback không phụ thuộc tqdm: trả về chính iterable, bỏ qua args."""
        return iterable if iterable is not None else range(0)


class TextClassificationDataset(Dataset):
    """
    Bọc 1 DataFrame có cột "text" (str) và "label" (int, đã map 0..C-1,
    xem data/loaders.py/data/poison.py) thành Dataset ĐÃ TOKENIZE SẴN
    (không còn tokenize-on-the-fly trong __getitem__, xem ghi chú đầu file).
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_length: int = 256,
        tokenize_batch_size: int = 1000,
        show_progress: Optional[bool] = None,
    ):
        """
        tokenize_batch_size: số dòng tokenize mỗi lần gọi tokenizer (gọi
            tokenizer theo batch luôn nhanh hơn nhiều so với gọi N lần cho
            N dòng đơn lẻ -- tokenizer "fast" tận dụng song song hoá nội bộ
            tốt hơn khi nhận list dài thay vì list từng-phần-tử-một).
        show_progress: None -> tự bật tqdm nếu DataFrame có >= 2000 dòng
            (đỡ làm rối log khi tokenize các batch nhỏ, vd D_i^trust hay
            D_poison chỉ vài trăm dòng); True/False để ép buộc bật/tắt.
        """
        # reset_index để __getitem__ luôn đánh chỉ số 0..len-1 liên tục,
        # bất kể index gốc của df (df thường là 1 lát cắt .loc[indices] từ
        # partition, nên index KHÔNG liên tục).
        df = df.reset_index(drop=True)
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.max_length = max_length

        n = len(self.texts)
        if show_progress is None:
            show_progress = n >= 2000

        all_input_ids: list[torch.Tensor] = []
        all_attention_mask: list[torch.Tensor] = []

        batch_ranges = range(0, n, tokenize_batch_size)
        iterator = (
            tqdm(batch_ranges, desc=f"Tokenizing {n} mẫu", unit="batch")
            if show_progress
            else batch_ranges
        )

        for start in iterator:
            end = min(start + tokenize_batch_size, n)
            chunk_texts = self.texts[start:end]
            encoding = tokenizer(
                chunk_texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            all_input_ids.append(encoding["input_ids"])
            all_attention_mask.append(encoding["attention_mask"])

        self.input_ids = torch.cat(all_input_ids, dim=0) if all_input_ids else torch.empty(0, dtype=torch.long)
        self.attention_mask = (
            torch.cat(all_attention_mask, dim=0) if all_attention_mask else torch.empty(0, dtype=torch.long)
        )
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        # Không còn gọi tokenizer ở đây -- chỉ index vào tensor đã tokenize
        # sẵn trong __init__ (xem ghi chú đầu file).
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels_tensor[idx],
        }


def make_dataloader(
    df,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    seed: Optional[int] = None,
    max_length: int = 256,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    tokenize_batch_size: int = 1000,
    show_tokenize_progress: Optional[bool] = None,
) -> DataLoader:
    """
    seed: seed CỤC BỘ cho việc xáo trộn minibatch, chỉ có ý nghĩa khi
    shuffle=True (Train/Fit). Validation/Test luôn shuffle=False, seed bị
    bỏ qua (không ảnh hưởng tới kết quả, chỉ ghi để rõ ràng lời gọi).

    num_workers: số tiến trình con dùng để load batch song song. Mặc định
        0 (chạy trên main process) để an toàn trên Colab/Jupyter, vì
        multiprocessing DataLoader (num_workers > 0) đôi khi gây lỗi hoặc
        treo trên notebook (đặc biệt nếu tokenizer/model không "pickle"
        được sạch sẽ). Vì __getitem__ giờ chỉ index tensor có sẵn (rất
        rẻ), num_workers > 0 KHÔNG còn quan trọng để tăng tốc như trước
        (khi __getitem__ còn gọi tokenizer) -- có thể thử num_workers=2
        để lợi thêm chút ít, nhưng không bắt buộc. Nếu gặp lỗi liên quan
        tới multiprocessing trên Colab, trả về 0 là lựa chọn an toàn nhất.
    pin_memory: None -> tự bật True nếu CUDA khả dụng (tăng tốc copy
        host->device nhẹ), False nếu không có CUDA hoặc muốn tắt tường minh.
    """
    dataset = TextClassificationDataset(
        df,
        tokenizer,
        max_length=max_length,
        tokenize_batch_size=tokenize_batch_size,
        show_progress=show_tokenize_progress,
    )
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
