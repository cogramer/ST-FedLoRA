"""
federated/aggregation/base.py

Tiện ích dùng chung cho mọi cơ chế tổng hợp (mục 3.1-3.4, 6.1-6.2 đề cương).

QUY ƯỚC VECTOR ĐẦU VÀO (mục 3.2, 3.4):
    u_i = Concat(vec(ΔA_i), vec(ΔB_i))

Việc flatten/unflatten một delta-state-dict (OrderedDict[str, Tensor] chỉ
chứa các tham số LoRA A/B) thành vector 1 chiều ĐÃ được cài trong
attack/cc_malicious_lora_sgd.py (flatten_delta_state_dict / unflatten_to_state_dict),
dùng sorted-key Concat(vec(ΔA),vec(ΔB)) và khớp thứ tự với
models/modernbert_lora.py::lora_state_dict_to_flat_vector.

=> Module này KHÔNG viết lại flatten/unflatten, mà import và tái sử dụng,
   đúng theo quyết định đã ghi trong tiến trình dự án. Mọi aggregator trong
   package này (fedavg.py, krum.py, trimmed_mean.py, tc_lora.py) chỉ thao
   tác trên OrderedDict[str, Tensor] (delta state dict) hoặc trên vector đã
   flatten, và dùng các hàm ở đây để chuyển qua lại.

Nếu tên hàm trong attack/cc_malicious_lora_sgd.py ở repo thực tế khác với
tên dưới đây, chỉ cần sửa đúng 1 chỗ trong phần import bên dưới.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Sequence, TypeAlias

import torch

# Tái sử dụng flatten/unflatten đã có (mục 3.2/3.4: u_i = Concat(vec(ΔA_i), vec(ΔB_i))).
from attack.cc_malicious_lora_sgd import (  # noqa: F401
    flatten_delta_state_dict,
    unflatten_to_state_dict,
)

StateDict: TypeAlias = Dict[str, torch.Tensor]


# --------------------------------------------------------------------------- #
# Số học cơ bản trên state dict (áp dụng công thức factor-wise mục 3.4)
# --------------------------------------------------------------------------- #

def zeros_like_state_dict(reference_sd: Dict[str, torch.Tensor]) -> StateDict:
    """Trả về một delta state dict toàn số 0, cùng shape/dtype/device với reference_sd."""
    return OrderedDict((k, torch.zeros_like(v)) for k, v in reference_sd.items())


def scale_state_dict(
    sd: Dict[str, torch.Tensor], scalar: float
) -> StateDict:
    """scalar * sd (theo từng tensor)."""
    return OrderedDict((k, v * scalar) for k, v in sd.items())


def add_state_dicts(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    scale_b: float = 1.0,
) -> StateDict:
    """sd_a + scale_b * sd_b, theo từng key (hai state dict phải cùng tập key)."""
    if sd_a.keys() != sd_b.keys():
        raise ValueError(
            "add_state_dicts: hai state dict không khớp tập key "
            f"({set(sd_a.keys()) ^ set(sd_b.keys())})."
        )
    return OrderedDict((k, sd_a[k] + scale_b * sd_b[k]) for k in sd_a.keys())


def weighted_average_state_dicts(
    deltas: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> StateDict:
    """
    Σ_i weight_i * delta_i, theo từng tensor (dùng cho FedAvg mục 3.1,
    Multi-Krum mục 3.2, Trimmed-Mean bản tensor-wise không dùng tới hàm này
    vì Trimmed Mean là theo TỪNG TOẠ ĐỘ, xem trimmed_mean.py).

    weights không bắt buộc tổng bằng 1 (hàm không tự chuẩn hoá); tầng gọi
    (fedavg.py, krum.py, tc_lora.py) chịu trách nhiệm chuẩn hoá theo đúng
    công thức của mình trước khi gọi hàm này.
    """
    if len(deltas) == 0:
        raise ValueError("weighted_average_state_dicts: danh sách deltas rỗng.")
    if len(deltas) != len(weights):
        raise ValueError(
            f"weighted_average_state_dicts: số deltas ({len(deltas)}) != số weights ({len(weights)})."
        )
    out = zeros_like_state_dict(deltas[0])
    for delta_i, w_i in zip(deltas, weights):
        out = add_state_dicts(out, delta_i, scale_b=float(w_i))
    return out


# --------------------------------------------------------------------------- #
# Chuyển list các delta state dict <-> ma trận vector phẳng [N, D]
# --------------------------------------------------------------------------- #

def stack_flat_vectors(
    deltas: Sequence[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """
    Flatten từng delta state dict thành vector 1-D (dùng flatten_delta_state_dict
    tái sử dụng từ attack/cc_malicious_lora_sgd.py), rồi xếp chồng thành ma trận
    [N, D] với N = số client, D = tổng số phần tử của (ΔA, ΔB).

    Đây chính là u_1, ..., u_N trong mục 3.2 (Krum) và 3.3 (Trimmed Mean).
    """
    if len(deltas) == 0:
        raise ValueError("stack_flat_vectors: danh sách deltas rỗng.")
    flat_vectors = [flatten_delta_state_dict(d) for d in deltas]
    return torch.stack(flat_vectors, dim=0)


def pairwise_sq_euclidean(vectors: torch.Tensor) -> torch.Tensor:
    """
    d(i,j) = ||u_i - u_j||_2^2 cho mọi cặp (i,j), mục 3.2.

    vectors: tensor [N, D]. Trả về ma trận [N, N], đường chéo = 0.
    Dùng công thức mở rộng ||u_i-u_j||^2 = ||u_i||^2 + ||u_j||^2 - 2<u_i,u_j>
    để tránh vòng lặp N^2 lần trừ vector D chiều (ổn định số học đủ dùng ở quy
    mô N=5, D nhỏ của đề tài).
    """
    sq_norms = (vectors * vectors).sum(dim=1)  # [N]
    gram = vectors @ vectors.t()  # [N, N]
    dist2 = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * gram
    # Khử sai số dấu phẩy động có thể ra số âm rất nhỏ ở đường chéo / cặp trùng nhau.
    dist2 = torch.clamp(dist2, min=0.0)
    return dist2


def cosine_similarity_matrix(
    candidates: torch.Tensor,
    references: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    s_{ij} = <u_j, d_i> / (||u_j||_2 * ||d_i||_2 + eps)   (mục 6.2 bước 3)

    candidates: [N, D] -- u_1..u_N (candidate adapter updates).
    references: [N, D] -- d_1^val..d_N^val (local validation descent direction
                của từng evaluator i).
    Trả về S [N, N] với S[i, j] = s_{ij} (hàng i = evaluator, cột j = candidate).

    Nếu ||u_j|| hoặc ||d_i|| ~ 0 (dưới eps), score tương ứng được đặt = 0
    đúng theo mô tả "Nếu một trong hai vector có norm gần bằng không, score
    được đặt bằng 0" (mục 6.2 bước 3).
    """
    if candidates.shape[1] != references.shape[1]:
        raise ValueError(
            "cosine_similarity_matrix: candidates và references khác số chiều "
            f"({candidates.shape[1]} != {references.shape[1]})."
        )
    cand_norm = candidates.norm(dim=1)  # [N] -- ||u_j||
    ref_norm = references.norm(dim=1)  # [N] -- ||d_i||

    dot = references @ candidates.t()  # [N, N], dot[i, j] = <d_i, u_j>
    denom = ref_norm.unsqueeze(1) * cand_norm.unsqueeze(0) + eps  # [N, N]
    scores = dot / denom

    # Mask: norm gần 0 (dưới eps, không phải dưới mức "bình thường") -> set 0.
    degenerate_cand = cand_norm <= eps  # [N] theo cột j
    degenerate_ref = ref_norm <= eps  # [N] theo hàng i
    scores = scores.masked_fill(degenerate_cand.unsqueeze(0), 0.0)
    scores = scores.masked_fill(degenerate_ref.unsqueeze(1), 0.0)
    return scores


def validate_byzantine_count(num_clients: int, f: int, *, context: str = "") -> None:
    """
    Kiểm tra điều kiện số lượng 2f+2 < N dùng cho Krum/Multi-Krum (mục 3.2).
    Không enforce cho Trimmed Mean (vốn chỉ cần N - 2*floor(beta*N) >= 1).
    """
    if f < 0:
        raise ValueError(f"{context}: f phải >= 0, nhận f={f}.")
    if 2 * f + 2 >= num_clients:
        raise ValueError(
            f"{context}: điều kiện 2f+2 < N không thỏa (f={f}, N={num_clients}). "
            "Theo mục 3.2, Krum/Multi-Krum yêu cầu 2f+2 < N."
        )
