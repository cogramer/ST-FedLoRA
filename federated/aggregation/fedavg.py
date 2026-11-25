"""
federated/aggregation/fedavg.py

FedAvg factor-wise cho LoRA (mục 3.1 và 3.4 đề cương).

    p_i = n_i^fit / n_fit
    θ_{t+1} = θ_t + Σ_i p_i * Δθ_{i,t}

Áp dụng factor-wise (không stacking kiểu FLoRA, xem mục 3.4):
    A_{t+1} = A_t + Σ_i p_i ΔA_{i,t}
    B_{t+1} = B_t + Σ_i p_i ΔB_{i,t}

Vì mọi aggregator trong package này đều trả về MỘT delta state dict tổng hợp
(không cộng trực tiếp vào A_t,B_t), hàm ở đây chỉ cần tính:
    Δθ_aggregated = Σ_i p_i * Δθ_i
Tầng gọi (federated/ main loop) sẽ cộng Δθ_aggregated vào θ_t.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from federated.aggregation.base import weighted_average_state_dicts

StateDict = "OrderedDict[str, torch.Tensor]"


def compute_sample_weights(num_samples: Sequence[int]) -> List[float]:
    """
    p_i = n_i^fit / n_fit , với n_fit = Σ_i n_i^fit  (mục 3.1).
    """
    if len(num_samples) == 0:
        raise ValueError("compute_sample_weights: danh sách num_samples rỗng.")
    n_fit = sum(num_samples)
    if n_fit <= 0:
        raise ValueError(f"compute_sample_weights: tổng n_fit phải > 0, nhận {n_fit}.")
    return [float(n_i) / float(n_fit) for n_i in num_samples]


def aggregate(
    client_deltas: Sequence[Dict[str, torch.Tensor]],
    num_samples: Sequence[int],
) -> Tuple["OrderedDict[str, torch.Tensor]", Dict]:
    """
    client_deltas[i]: delta state dict của Client i (Δθ_{i,t} = ΔA_i ⊕ ΔB_i).
    num_samples[i]:   n_i^fit tương ứng (số mẫu D_i^fit dùng để huấn luyện cục bộ).

    Trả về (aggregated_delta, diagnostics). diagnostics chứa trọng số p_i đã dùng,
    để ghi log / đối chiếu với mục 8 bảng tham số.
    """
    if len(client_deltas) != len(num_samples):
        raise ValueError(
            "fedavg.aggregate: số client_deltas "
            f"({len(client_deltas)}) != số num_samples ({len(num_samples)})."
        )
    weights = compute_sample_weights(num_samples)
    aggregated = weighted_average_state_dicts(client_deltas, weights)
    diagnostics = {
        "mechanism": "fedavg",
        "weights": weights,
        "num_samples": list(num_samples),
    }
    return aggregated, diagnostics
