"""
federated/aggregation/trimmed_mean.py

Coordinate-wise Trimmed Mean (mục 3.3 đề cương, gốc Yin et al. 2018).

Với mỗi toạ độ k, sắp xếp N giá trị:
    u_(1),k <= u_(2),k <= ... <= u_(N),k

Với tỷ lệ cắt beta:
    TrimMean_beta({u_i})_k = 1/((1-2*beta)*N) * Σ_{j=floor(beta*N)+1}^{N-floor(beta*N)} u_(j),k

Cấu hình chính của đề cương: N=5, f=1, beta=f/N=0.2 -> floor(beta*N)=1,
mỗi toạ độ loại 1 giá trị nhỏ nhất + 1 giá trị lớn nhất, lấy trung bình
3 giá trị còn lại:
    u_TM,k = (u_(2),k + u_(3),k + u_(4),k) / 3

Trimmed Mean không tạo nhãn benign/Byzantine cho từng Client (nó hoạt động
theo TỪNG TOẠ ĐỘ độc lập) -- đúng ghi chú mục 7.4, nên diagnostics ở đây
không có selected_index/selected_indices như Krum/Multi-Krum.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch

from federated.aggregation.base import stack_flat_vectors, unflatten_to_state_dict

StateDict = "OrderedDict[str, torch.Tensor]"


def aggregate(
    client_deltas: Sequence[Dict[str, torch.Tensor]],
    beta: float,
) -> Tuple["OrderedDict[str, torch.Tensor]", Dict]:
    """
    client_deltas: danh sách N delta state dict (ΔA_i ⊕ ΔB_i).
    beta: tỷ lệ cắt mỗi đầu (đề cương cố định beta = f/N = 0.2 cho N=5,f=1).

    Trả về (aggregated_delta, diagnostics).
    """
    vectors = stack_flat_vectors(client_deltas)  # [N, D]
    num_clients = vectors.shape[0]

    if not (0.0 <= beta < 0.5):
        raise ValueError(f"trimmed_mean.aggregate: beta phải trong [0, 0.5), nhận beta={beta}.")

    k = math.floor(beta * num_clients)  # floor(beta*N)
    kept_count = num_clients - 2 * k
    if kept_count < 1:
        raise ValueError(
            f"trimmed_mean.aggregate: beta={beta} với N={num_clients} khiến "
            f"số giá trị còn lại sau khi cắt = {kept_count} < 1. Giảm beta hoặc tăng N."
        )

    # Sắp xếp theo từng toạ độ (dim=0 ứng với N client).
    sorted_vectors, _ = torch.sort(vectors, dim=0)  # [N, D]

    # Giữ lại các hàng từ chỉ số k (0-based) tới N-k-1 (0-based), tức
    # 1-based j = k+1 .. N-k, đúng công thức Σ_{j=floor(beta*N)+1}^{N-floor(beta*N)}.
    kept = sorted_vectors[k : num_clients - k, :]  # [kept_count, D]

    averaged_vector = kept.mean(dim=0)  # 1/((1-2beta)N) * Σ kept, vì kept_count=(1-2beta)N
    aggregated = unflatten_to_state_dict(averaged_vector, client_deltas[0])

    diagnostics = {
        "mechanism": "trimmed_mean",
        "beta": beta,
        "trimmed_each_side": k,
        "kept_count": kept_count,
    }
    return aggregated, diagnostics
