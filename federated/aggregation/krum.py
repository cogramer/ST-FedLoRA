"""
federated/aggregation/krum.py

Krum và Multi-Krum (mục 3.2 đề cương, gốc Blanchard et al. 2017).

Vector đầu vào (mục 3.2):
    u_i = Concat(vec(ΔA_i), vec(ΔB_i))

Với mỗi cặp i != j:
    d(i,j) = ||u_i - u_j||_2^2

Gọi N_i là tập chỉ số của N-f-2 vector GẦN u_i NHẤT (không tính chính u_i):
    Score(i) = Σ_{j in N_i} ||u_i - u_j||_2^2
    i* = argmin_i Score(i)
    u_Krum = u_{i*}

Multi-Krum: chọn m vector có Score thấp nhất, lấy trung bình:
    M_m = tập gồm m chỉ số có Score nhỏ nhất
    u_MultiKrum = (1/m) * Σ_{i in M_m} u_i

Điều kiện số lượng (định lý Krum gốc): 2f+2 < N. Với cấu hình đề cương
N=5, f=1 -> 4<5 thỏa ở biên, N-f-2=2 láng giềng mỗi vector. Mặc định của
module là m = N-f cho Multi-Krum, đúng cấu hình chính m=N-f=4 đã chọn,
nhưng vẫn cho phép truyền m tường minh để phục vụ ablation (mục 7.5).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

from federated.aggregation.base import (
    pairwise_sq_euclidean,
    stack_flat_vectors,
    unflatten_to_state_dict,
    validate_byzantine_count,
    StateDict,
)


def _krum_scores(vectors: torch.Tensor, f: int) -> torch.Tensor:
    """
    Tính Score(i) cho mọi i, dựa trên N-f-2 láng giềng gần nhất (không tính
    chính nó). Trả về tensor [N] các điểm Score.
    """
    num_clients = vectors.shape[0]
    validate_byzantine_count(num_clients, f, context="krum._krum_scores")

    num_neighbors = num_clients - f - 2  # N-f-2, luôn >= 1 vì đã validate 2f+2<N
    dist2 = pairwise_sq_euclidean(vectors)  # [N, N], đường chéo = 0

    # Loại trừ chính nó khỏi việc chọn láng giềng gần nhất: đặt khoảng cách
    # tới chính nó = +inf trước khi lấy top-k nhỏ nhất.
    dist2_excl_self = dist2.clone()
    diag_idx = torch.arange(num_clients)
    dist2_excl_self[diag_idx, diag_idx] = float("inf")

    # topk nhỏ nhất (num_neighbors) theo mỗi hàng -> sum lại thành Score(i).
    nearest, _ = torch.topk(dist2_excl_self, k=num_neighbors, dim=1, largest=False)
    scores = nearest.sum(dim=1)  # [N]
    return scores


def aggregate_krum(
    client_deltas: Sequence[Dict[str, torch.Tensor]],
    f: int,
) -> Tuple[StateDict, Dict]:
    """
    Krum: chọn đúng MỘT client (vector nguyên bản, không pha trộn).

    Trả về (delta của client được chọn, diagnostics). diagnostics gồm:
      - selected_index: chỉ số i* trong client_deltas (0-based, theo thứ tự
        truyền vào, KHÔNG phải client_id tuỳ ý -- tầng gọi tự ánh xạ).
      - scores: Score(i) của mọi client (để thống kê tần suất được chọn
        qua các round, mục 7.4: "với Krum/Multi-Krum, có thể báo cáo tần
        suất lựa chọn của từng Client").
    """
    vectors = stack_flat_vectors(client_deltas)  # [N, D]
    scores = _krum_scores(vectors, f)
    selected_index = int(torch.argmin(scores).item())

    # Nếu có đồng hạng (hiếm khi xảy ra với dữ liệu thực, nhưng torch.argmin
    # đã tự chọn chỉ số nhỏ nhất đầu tiên trong trường hợp bằng điểm -- khớp
    # quy ước "nếu bằng điểm, chọn identifier nhỏ nhất" của Krum gốc).
    aggregated = client_deltas[selected_index]
    diagnostics = {
        "mechanism": "krum",
        "f": f,
        "num_neighbors": vectors.shape[0] - f - 2,
        "selected_index": selected_index,
        "scores": scores.tolist(),
    }
    return aggregated, diagnostics


def aggregate_multi_krum(
    client_deltas: Sequence[Dict[str, torch.Tensor]],
    f: int,
    m: int | None = None,
) -> Tuple[StateDict, Dict]:
    """
    Multi-Krum: chọn m vector có Score thấp nhất, lấy trung bình cộng
    (không trọng số theo mẫu -- đúng công thức mục 3.2, khác FedAvg).

    m mặc định = N - f (cấu hình chính của đề cương, N=5,f=1 -> m=4).
    """
    vectors = stack_flat_vectors(client_deltas)  # [N, D]
    num_clients = vectors.shape[0]
    scores = _krum_scores(vectors, f)

    if m is None:
        actual_m = num_clients - f
    else:
        actual_m = m

    if not (1 <= actual_m <= num_clients):
        raise ValueError(
            f"aggregate_multi_krum: m phải trong [1, N] "
            f"(N={num_clients}), nhận m={actual_m}."
        )

    # m chỉ số có Score nhỏ nhất.
    _, selected_indices = torch.topk(scores, k=actual_m, largest=False)
    selected_indices = selected_indices.tolist()

    selected_vectors = vectors[selected_indices]  # [m, D]
    averaged_vector = selected_vectors.mean(dim=0)  # (1/m) Σ u_i

    aggregated = unflatten_to_state_dict(averaged_vector, client_deltas[0])
    diagnostics = {
        "mechanism": "multi_krum",
        "f": f,
        "m": actual_m,
        "selected_indices": selected_indices,
        "scores": scores.tolist(),
    }
    return aggregated, diagnostics
