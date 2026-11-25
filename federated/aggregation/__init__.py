"""
federated/aggregation

Package tổng hợp cho HFL/LoRA (mục 3.1-3.4, 6.1-6.2 đề cương). Mỗi round,
Server nhận N delta state dict (ΔA_i ⊕ ΔB_i) từ N Client trên CÙNG một
trạng thái global (A_t, B_t) đầu round, rồi chạy ĐỘC LẬP từng cơ chế dưới
đây trên cùng một bộ client submissions đó (mục 7.1 bước 4):

    FedAvg        -> fedavg.aggregate
    Krum          -> krum.aggregate_krum
    Multi-Krum    -> krum.aggregate_multi_krum
    Trimmed Mean  -> trimmed_mean.aggregate
    TC-LoRA       -> tc_lora.aggregate_from_scores

Mọi hàm aggregate_* trả về (aggregated_delta, diagnostics), với
aggregated_delta là một delta state dict (ΔA ⊕ ΔB) để tầng gọi tự cộng
vào (A_t, B_t):

    A_{t+1}, B_{t+1} = tách aggregated_delta rồi A_t+ΔA, B_t+ΔB

(Riêng TC-LoRA có thể trả aggregated_delta=None khi round bị bỏ --
xem tc_lora.aggregate_from_scores.)

run_aggregator() bên dưới là một dispatcher mỏng cho tiện orchestration,
KHÔNG ép các cơ chế về chung một chữ ký tham số (vì TC-LoRA cần
score_matrix đã tính sẵn từ model thật ở tầng federated/, trong khi
FedAvg/Krum/Trimmed-Mean chỉ cần client_deltas + vài con số cấu hình).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch

from federated.aggregation import fedavg, krum, tc_lora, trimmed_mean

StateDict = "OrderedDict[str, torch.Tensor]"

SUPPORTED_MECHANISMS = ("fedavg", "krum", "multi_krum", "trimmed_mean", "tc_lora")


def run_aggregator(
    mechanism: str,
    client_deltas: Sequence[Dict[str, torch.Tensor]],
    **kwargs: Any,
) -> Tuple[Any, Dict]:
    """
    Dispatcher tiện dụng. Tham số bắt buộc theo từng mechanism:

        "fedavg":        num_samples: Sequence[int]
        "krum":          f: int
        "multi_krum":    f: int, m: Optional[int] = None (mặc định N-f)
        "trimmed_mean":  beta: float
        "tc_lora":       score_matrix: torch.Tensor [N, N] (đã tính sẵn ở
                         federated/ main loop qua tc_lora.consistency_matrix,
                         dùng d_i^val THẬT từ model+D_i^trust)
                         eps: float = 1e-8 (tuỳ chọn)

    Trả về (aggregated_delta_hoặc_None, diagnostics) y hệt hàm con tương ứng.
    """
    if mechanism == "fedavg":
        return fedavg.aggregate(client_deltas, num_samples=kwargs["num_samples"])

    if mechanism == "krum":
        return krum.aggregate_krum(client_deltas, f=kwargs["f"])

    if mechanism == "multi_krum":
        return krum.aggregate_multi_krum(
            client_deltas, f=kwargs["f"], m=kwargs.get("m")
        )

    if mechanism == "trimmed_mean":
        return trimmed_mean.aggregate(client_deltas, beta=kwargs["beta"])

    if mechanism == "tc_lora":
        return tc_lora.aggregate_from_scores(
            client_deltas,
            score_matrix=kwargs["score_matrix"],
            eps=kwargs.get("eps", 1e-8),
        )

    raise ValueError(
        f"run_aggregator: mechanism '{mechanism}' không được hỗ trợ. "
        f"Các giá trị hợp lệ: {SUPPORTED_MECHANISMS}."
    )


__all__ = [
    "fedavg",
    "krum",
    "trimmed_mean",
    "tc_lora",
    "run_aggregator",
    "SUPPORTED_MECHANISMS",
]
