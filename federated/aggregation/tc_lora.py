"""
federated/aggregation/tc_lora.py

TC-LoRA: Trust-Consistency Weighted LoRA Aggregation (mục 6.2 đề cương),
lấy cảm hứng consistency-based trust score từ TrustGraph-DFL [21] nhưng
tổng hợp TẬP TRUNG tại Server cho HFL (không phải peer-to-peer).

QUAN TRỌNG VỀ PHẠM VI MODULE NÀY:
    Bước 2 của mục 6.2 ("Tính local validation descent direction") cần
    forward + backward pass THẬT trên model + D_i^trust của từng Client
    (d_{i,t}^val = -∇_{(A,B)} L_i^val(A_t,B_t;H*)). Việc đó đụng tới model
    thật và dữ liệu thật, nên PHẢI nằm ở tầng orchestration (federated/
    main loop: mỗi Client tự tính d_i^val cục bộ, Server không thấy
    D_i^trust thô -- đúng threat model mục 6.2 bước 6).

    Module này chỉ cài hoá các phép biến đổi THUẦN VECTOR/MA TRẬN ở bước
    3-5 (sau khi mỗi Client ĐÃ gửi về Server các scalar score):
        bước 3: s_{ij} = <u_j, d_i^val> / (||u_j|| ||d_i^val|| + eps)
        bước 4: q_j = max(0, Median_i s_{ij}); w_j = q_j / Σ_k q_k
        bước 5: A_{t+1} = A_t + Σ_j w_j ΔA_j ; B_{t+1} = B_t + Σ_j w_j ΔB_j

    consistency_matrix() ở dưới có thể gọi được thuần bằng vector (không
    cần model), nên được đặt ở đây để tiện unit-test độc lập; nhưng việc
    tạo ra d_i^val từ model+D_i^trust vẫn là việc của federated/ main loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from federated.aggregation.base import (
    cosine_similarity_matrix,
    stack_flat_vectors,
    weighted_average_state_dicts,
)

StateDict = "OrderedDict[str, torch.Tensor]"


def consistency_matrix(
    candidate_deltas: Sequence[Dict[str, torch.Tensor]],
    validation_directions: Sequence[torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Bước 3 (mục 6.2): dựng ma trận consistency score S[i, j] = s_{ij}, với
    i = evaluator (Client đánh giá), j = candidate (Client có update được
    đánh giá).

    candidate_deltas: danh sách N delta state dict u_1..u_N (ΔA_j ⊕ ΔB_j),
        theo ĐÚNG THỨ TỰ Client 1..N -- đây chính là các candidate mà Server
        phân phối cho từng Client đánh giá (mục 6.2 bước 6).
    validation_directions: danh sách N vector đã flatten d_1^val..d_N^val
        (mỗi Client i tự tính d_i^val = -∇L_i^val cục bộ rồi gửi dạng đã
        flatten về Server, hoặc Server tự flatten từ state dict Client gửi
        -- quy ước flatten phải khớp flatten_delta_state_dict dùng cho
        candidate_deltas để hai phía cùng không gian toạ độ).

    Trả về S kích thước [N, N].
    """
    if len(candidate_deltas) != len(validation_directions):
        raise ValueError(
            "consistency_matrix: số candidate_deltas "
            f"({len(candidate_deltas)}) != số validation_directions "
            f"({len(validation_directions)})."
        )
    candidates = stack_flat_vectors(candidate_deltas)  # [N, D], u_1..u_N
    references = torch.stack(list(validation_directions), dim=0)  # [N, D], d_1^val..d_N^val
    return cosine_similarity_matrix(candidates, references, eps=eps)


def compute_trust_weights(
    score_matrix: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[Optional[torch.Tensor], Dict]:
    """
    Bước 4 (mục 6.2):
        q_j = max(0, Median_{i in {1,...,N}} s_{ij})
        w_j = q_j / Σ_k q_k   (nếu Σ_k q_k > eps)

    score_matrix: [N, N], S[i, j] = evaluator i chấm điểm candidate j.

    Trả về (weights hoặc None, diagnostics).
    weights là tensor [N] (w_1..w_N) nếu tổng q_k > eps; nếu không, trả về
    None để báo "round không có cập nhật được chấp nhận theo TC-LoRA" --
    tầng gọi (aggregate_from_scores / federated main loop) phải GIỮ NGUYÊN
    (A_t, B_t) ở round đó, đúng mô tả cuối bước 4.
    """
    # Median theo chiều evaluator (dim=0, tức theo từng cột j).
    median_scores, _ = torch.median(score_matrix, dim=0)  # [N]
    q = torch.clamp(median_scores, min=0.0)  # max(0, Median_i s_ij)

    total_q = q.sum()
    diagnostics = {
        "mechanism": "tc_lora",
        "median_scores": median_scores.tolist(),
        "q": q.tolist(),
        "total_q": float(total_q.item()),
    }

    if float(total_q.item()) <= eps:
        diagnostics["round_accepted"] = False
        diagnostics["weights"] = None
        return None, diagnostics

    weights = q / total_q
    diagnostics["round_accepted"] = True
    diagnostics["weights"] = weights.tolist()
    return weights, diagnostics


def aggregate_from_scores(
    client_deltas: Sequence[Dict[str, torch.Tensor]],
    score_matrix: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[Optional["OrderedDict[str, torch.Tensor]"], Dict]:
    """
    Gộp bước 4-5 (mục 6.2): từ ma trận score (đã có sẵn, do federated/ main
    loop thu thập từ các Client), tính trọng số trust w_j rồi tổng hợp
    adapter factor-wise:
        A_{t+1} = A_t + Σ_j w_j ΔA_j ,  B_{t+1} = B_t + Σ_j w_j ΔB_j

    Trả về (aggregated_delta, diagnostics). aggregated_delta = None nếu
    round bị bỏ (Σ q_k <= eps) -- tầng gọi tự quyết định giữ nguyên
    (A_t, B_t) khi gặp None, đúng mô tả cuối bước 4 của đề cương.
    """
    weights, diagnostics = compute_trust_weights(score_matrix, eps=eps)
    if weights is None:
        return None, diagnostics

    aggregated = weighted_average_state_dicts(client_deltas, weights.tolist())
    return aggregated, diagnostics


def flag_suspicious_clients(
    weights: Sequence[float],
    gamma_trust: float,
) -> List[bool]:
    """
    Tiện ích cho mục 7.4 (TPR/FPR của TC-LoRA): đánh dấu Client j "nghi vấn"
    khi w_j < gamma_trust.

    Đây CHỈ là phép so sánh ngưỡng thuần tuý trên vector trọng số đã có
    (không tự chọn gamma_trust -- giá trị này phải được chọn trên
    Validation hoặc cố định trước khi xem Test, theo đúng mục 7.4).
    Việc đếm TP/FP/TN/FN qua nhiều round/seed là việc của eval/, không
    lặp lại ở đây.

    Trả về list[bool] cùng độ dài weights, True nghĩa là "bị đánh dấu
    nghi vấn" (w_j < gamma_trust).
    """
    return [float(w_j) < float(gamma_trust) for w_j in weights]
