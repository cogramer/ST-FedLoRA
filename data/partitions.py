"""
data/partition.py
Chia dữ liệu huấn luyện cho K Client theo 2 kịch bản: IID và Non-IID
(Dirichlet), đúng theo mục 4.4 trong đề cương.

Theo mục 5 thì mọi hàm ở đây dùng
`np.random.default_rng(seed)` cục bộ — KHÔNG đụng vào global
`np.random.seed()` / `torch.manual_seed()`, Lý do vì toàn hệ thống có 3 nguồn
ngẫu nhiên phải tách biệt hoàn toàn để đảm bảo tính reproducibility độc lập:
    1. Khởi tạo ma trận LoRA A (Gaussian)      -> seed riêng ở model init
    2. Chia dữ liệu Non-IID (Dirichlet)        -> seed riêng ở module này
    3. Nhiễu Gaussian của attacker (ξ)         -> seed riêng ở attack module
Nếu dùng chung 1 global RNG thì thứ tự gọi hàm giữa các module sẽ gây lệch kết
quả dù cùng "seed" tổng. Chỉ nhận seed qua tham số, không bao giờ gọi `np.random.seed()`.

Module này chỉ quyết định "mẫu nào thuộc về Client nào". Việc đầu độc dữ
liệu (chèn trigger, đổi nhãn target) cho Client độc hại là trách nhiệm của
`data/poison.py` (áp dụng sau khi đã có phân vùng ở đây).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def partition_iid(num_samples: int, num_clients: int, seed: int) -> list[np.ndarray]:
    """
    Xáo trộn ngẫu nhiên rồi chia đều `num_samples` index cho `num_clients`.
    Phần dư (num_samples % num_clients) được rải đều cho các client đầu tiên
    để chênh lệch số mẫu giữa các client tối đa là 1.
    """
    if num_clients < 1:
        raise ValueError("num_clients phải >= 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)

    base = num_samples // num_clients
    remainder = num_samples % num_clients

    partitions: list[np.ndarray] = []
    cursor = 0
    for k in range(num_clients):
        size = base + (1 if k < remainder else 0)
        partitions.append(indices[cursor:cursor + size])
        cursor += size
    return partitions


def partition_dirichlet(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
    max_retries: int = 100,
) -> list[np.ndarray]:
    """
    Phân phối Dirichlet theo từng lớp (chuẩn NIID-Bench / FedAvg Non-IID
    literature): với mỗi lớp c, lấy tỉ lệ p_k ~ Dir(alpha) cho K client rồi
    chia các mẫu thuộc lớp c theo tỉ lệ đó. alpha càng nhỏ -> lệch càng mạnh
    (mỗi client chỉ có vài lớp chiếm ưu thế); alpha càng lớn -> càng gần IID.

    `min_samples_per_client`: nếu sau khi chia có client nào nhận ít hơn
    ngưỡng này thì sẽ tự động thử lại (re-sample Dirichlet) tối đa `max_retries`
    lần. Tránh trường hợp 1 client "trắng tay" vì alpha quá nhỏ với dataset
    nhiều lớp (ví dụ như DBpedia 9 lớp, K=5).

    Output: list gồm K mảng index (ứng với vị trí trong `labels`) đã được
    xáo trộn trong nội bộ từng client.
    """
    if num_clients < 1:
        raise ValueError("num_clients phải >= 1")
    if alpha <= 0:
        raise ValueError("alpha (Dirichlet) phải > 0")

    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    rng = np.random.default_rng(seed)

    for attempt in range(1, max_retries + 1):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            class_idx = np.where(labels == c)[0]
            rng.shuffle(class_idx)

            # tỉ lệ chia lớp c cho K client
            proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
            # chuyển tỉ lệ -> số lượng mẫu (cumulative split points)
            split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
            class_splits = np.split(class_idx, split_points)

            for k in range(num_clients):
                client_indices[k].extend(class_splits[k].tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_samples_per_client:
            partitions = [rng.permutation(np.asarray(ci)) for ci in client_indices]
            return partitions

        continue

    raise RuntimeError(
        f"Không thể chia Dirichlet(alpha={alpha}) sao cho mỗi client có "
        f">= {min_samples_per_client} mẫu sau {max_retries} lần thử. "
        f"Kích thước client nhỏ nhất gần nhất: {min(sizes)}. "
        f"Hãy tăng alpha hoặc giảm min_samples_per_client."
    )


def partition_dataset(
    labels: np.ndarray,
    num_clients: int,
    scheme: str,
    seed: int,
    dirichlet_alpha: float = 0.5,
) -> list[np.ndarray]:
    """
    scheme: "iid" -> partition_iid
            "noniid" -> partition_dirichlet (Dir(dirichlet_alpha))
    """
    scheme = scheme.lower()
    if scheme == "iid":
        return partition_iid(num_samples=len(labels), num_clients=num_clients, seed=seed)
    elif scheme == "noniid":
        return partition_dirichlet(
            labels=np.asarray(labels),
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            seed=seed,
        )
    else:
        raise ValueError(f"scheme phải là 'iid' hoặc 'noniid', nhận '{scheme}'")


# Tiện ích kiểm tra phân bố để sanity-check trước khi train
def partition_summary(
    partitions: list[np.ndarray],
    labels: np.ndarray,
    label_names: Optional[dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Trả về bảng (client x class) đếm số mẫu mỗi client sở hữu theo từng lớp.
    Dùng để in ra kiểm tra trực quan mức độ lệch Non-IID trước khi chạy thật.
    """
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    cols = [label_names[c] if label_names else str(c) for c in range(num_classes)]

    rows = []
    for k, idx in enumerate(partitions):
        counts = np.bincount(labels[idx], minlength=num_classes)
        rows.append(counts)

    df = pd.DataFrame(rows, columns=cols)
    df.index = [f"client_{k}" for k in range(len(partitions))]
    df["total"] = df.sum(axis=1)
    return df


if __name__ == "__main__":
    # Mô phỏng nhanh với nhãn giả để kiểm tra tính nhất quán
    rng = np.random.default_rng(0)
    fake_labels = rng.integers(0, 9, size=24_0000 // 100)  # giả lập cỡ nhỏ

    print("=== IID ===")
    iid_parts = partition_iid(len(fake_labels), num_clients=5, seed=42)
    print(partition_summary(iid_parts, fake_labels))

    print("\n=== Non-IID Dirichlet(alpha=0.5) ===")
    noniid_parts = partition_dirichlet(fake_labels, num_clients=5, alpha=0.5, seed=42)
    print(partition_summary(noniid_parts, fake_labels))

    # Kiểm tra reproducibility: cùng seed -> cùng kết quả
    noniid_parts_2 = partition_dirichlet(fake_labels, num_clients=5, alpha=0.5, seed=42)
    same = all(np.array_equal(a, b) for a, b in zip(noniid_parts, noniid_parts_2))
    print(f"\nReproducible với cùng seed: {same}")