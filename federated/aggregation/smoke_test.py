"""
federated/aggregation/smoke_test.py

Script chạy thử nhanh (demo + sanity check) cho federated/aggregation/,
dùng được ngay trên Colab -- nơi đã có:
    - attack/cc_malicious_lora_sgd.py (flatten_delta_state_dict, unflatten_to_state_dict)
    - torch

KHÔNG dùng model/dữ liệu thật: chỉ dựng state dict LoRA (ΔA, ΔB) giả lập
đúng kiểu đặt tên tham số của models/modernbert_lora.py, để test ĐỘC LẬP
5 cơ chế tổng hợp (mục 3.1-3.4, 6.1-6.2 đề cương) trước khi ráp vào
federated/ main loop thật (còn chưa viết).

Chạy từ thư mục GỐC repo (nơi có package attack/, federated/, models/):
    python -m federated.aggregation.smoke_test

Nếu attack/cc_malicious_lora_sgd.py chưa nằm đúng vị trí đó trong repo thật,
sửa PYTHONPATH hoặc chạy từ đúng thư mục gốc chứa cả 2 package.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from federated.aggregation import SUPPORTED_MECHANISMS, run_aggregator
from federated.aggregation import fedavg, krum, tc_lora, trimmed_mean
from federated.aggregation.base import flatten_delta_state_dict, validate_byzantine_count, StateDict, unflatten_to_state_dict


def make_fake_lora_delta(scale: float, seed: int, shapes=None) -> StateDict:
    """
    Dựng 1 delta state dict (ΔA ⊕ ΔB) giả lập cho 1 layer LoRA của 1 Client,
    dùng tên tham số kiểu "...Wqkv.lora_A.weight"/"...Wqkv.lora_B.weight"
    (khớp mục 4.3.2: target_modules=["Wqkv"]) chỉ để demo cho thực tế --
    flatten_delta_state_dict không phụ thuộc vào tên cụ thể, chỉ cần sorted
    theo key là đủ.
    """
    if shapes is None:
        shapes = {
            "base_model.model.layers.0.attn.Wqkv.lora_A.weight": (4, 8),
            "base_model.model.layers.0.attn.Wqkv.lora_B.weight": (8, 4),
        }
    g = torch.Generator().manual_seed(seed)
    out: StateDict = OrderedDict()
    for name, shape in shapes.items():
        out[name] = torch.randn(shape, generator=g) * scale
    return out


def _local_randn_like(ref: torch.Tensor, seed: int, scale: float = 1.0) -> torch.Tensor:
    """
    torch.randn_like dùng RNG global, trái với quy ước "không bao giờ set
    seed global trong pipeline" của dự án -- hàm này thay thế bằng
    torch.Generator cục bộ theo đúng seed truyền vào, để toàn bộ script demo
    tái lập được 100% giữa các lần chạy (không phụ thuộc trạng thái RNG global).
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(ref.shape, generator=g) * scale


def main() -> None:
    N, f = 5, 1

    # 4 Client benign (scale nhỏ, gần nhau) + 1 Client Byzantine (scale lớn, lệch xa).
    client_deltas = [make_fake_lora_delta(scale=0.05, seed=100 + i) for i in range(4)]
    byzantine_delta = make_fake_lora_delta(scale=5.0, seed=999)
    client_deltas.append(byzantine_delta)
    byzantine_index = N - 1

    num_samples = [100, 150, 120, 130, 90]

    print("=== validate_byzantine_count ===")
    validate_byzantine_count(N, f, context="smoke_test")
    try:
        validate_byzantine_count(5, 2, context="smoke_test (phải lỗi, 2f+2=6 >= N=5)")
        raise AssertionError("Lẽ ra phải raise khi 2f+2>=N (f=2, N=5 -> 6>=5)")
    except ValueError as e:
        print("OK, raise đúng như kỳ vọng:", e)

    print("\n=== FedAvg (mục 3.1) ===")
    agg, diag = fedavg.aggregate(client_deltas, num_samples)
    assert abs(sum(diag["weights"]) - 1.0) < 1e-6
    for k, v in agg.items():
        assert v.shape == client_deltas[0][k].shape
    print("weights p_i:", diag["weights"])
    print("OK: shape giữ nguyên, tổng trọng số p_i = 1")

    print("\n=== Krum (mục 3.2) ===")
    agg, diag = krum.aggregate_krum(client_deltas, f=f)
    print("selected_index:", diag["selected_index"], "scores:", diag["scores"])
    assert diag["selected_index"] != byzantine_index, "Krum không được chọn Byzantine trong demo này"
    print("OK: Krum loại Byzantine")

    print("\n=== Multi-Krum (mục 3.2, m=N-f) ===")
    agg, diag = krum.aggregate_multi_krum(client_deltas, f=f)
    print("selected_indices:", diag["selected_indices"], "m:", diag["m"])
    assert byzantine_index not in diag["selected_indices"]
    print("OK: Multi-Krum loại Byzantine khỏi tập trung bình")

    print("\n=== Trimmed Mean (mục 3.3, beta=f/N=0.2) ===")
    agg, diag = trimmed_mean.aggregate(client_deltas, beta=f / N)
    print("trimmed_each_side:", diag["trimmed_each_side"], "kept_count:", diag["kept_count"])
    for k, v in agg.items():
        assert v.shape == client_deltas[0][k].shape
    print("OK: shape giữ nguyên sau Trimmed Mean")

    print("\n=== TC-LoRA (mục 6.2 bước 3-5) ===")
    # Giả lập d_i^val: 4 evaluator benign có hướng "giống" hướng benign trung bình
    # (cộng nhiễu nhỏ để không hoàn toàn trùng), 1 evaluator Byzantine có hướng
    # không tương quan (ngẫu nhiên độc lập) -- MÔ PHỎNG việc Client Byzantine
    # đánh giá lệch khỏi đồng thuận số đông, không giả định dấu cụ thể.
    benign_mean_delta, _ = fedavg.aggregate(client_deltas[:4], [1, 1, 1, 1])
    benign_dir = flatten_delta_state_dict(benign_mean_delta)

    validation_directions = []
    for i in range(4):
        noise = _local_randn_like(benign_dir, seed=2000 + i, scale=0.01)
        validation_directions.append(benign_dir + noise)
    # evaluator Byzantine: hướng không tương quan, seed riêng cố định.
    validation_directions.append(_local_randn_like(benign_dir, seed=2999, scale=1.0))

    score_matrix = tc_lora.consistency_matrix(client_deltas, validation_directions)
    print("score_matrix (hàng=evaluator i, cột=candidate j):\n", score_matrix)

    weights, diag = tc_lora.compute_trust_weights(score_matrix)
    assert weights is not None, "Demo này kỳ vọng round được chấp nhận (tổng q > eps)"
    print("weights w_j:", diag["weights"])
    assert abs(sum(diag["weights"]) - 1.0) < 1e-5
    assert diag["weights"][byzantine_index] < max(diag["weights"][:4]), (
        "Kỳ vọng trọng số trust của Byzantine thấp hơn ít nhất 1 Client benign trong demo này"
    )
    print("OK: TC-LoRA hạ trọng số Byzantine so với Client benign tốt nhất")

    aggregated, diag2 = tc_lora.aggregate_from_scores(client_deltas, score_matrix)
    assert aggregated is not None
    for k, v in aggregated.items():
        assert v.shape == client_deltas[0][k].shape
    print("OK: TC-LoRA aggregate_from_scores trả delta đúng shape")

    print("\n=== run_aggregator dispatcher (cả 5 cơ chế qua 1 cổng gọi chung) ===")
    assert set(SUPPORTED_MECHANISMS) == {"fedavg", "krum", "multi_krum", "trimmed_mean", "tc_lora"}
    _, d1 = run_aggregator("fedavg", client_deltas, num_samples=num_samples)
    _, d2 = run_aggregator("krum", client_deltas, f=f)
    _, d3 = run_aggregator("multi_krum", client_deltas, f=f)
    _, d4 = run_aggregator("trimmed_mean", client_deltas, beta=f / N)
    _, d5 = run_aggregator("tc_lora", client_deltas, score_matrix=score_matrix)
    for d in (d1, d2, d3, d4, d5):
        assert "mechanism" in d
    try:
        run_aggregator("khong_ton_tai", client_deltas)
        raise AssertionError("Lẽ ra phải raise ValueError cho mechanism không hợp lệ")
    except ValueError:
        pass
    print("OK: run_aggregator dispatch đúng cho cả 5 cơ chế + raise đúng khi tên sai")

    print("\n=== TC-LoRA: minh hoạ round bị bỏ (None) khi đồng thuận trust <= 0 ===")
    # Test trực tiếp compute_trust_weights với 1 score_matrix dựng thủ công
    # (không qua consistency_matrix/vector ngẫu nhiên) để tránh phụ thuộc vào
    # tương quan ngẫu nhiên không kiểm soát được giữa các hướng -- mọi cột j
    # đều có Median_i s_ij < 0 => q_j=0 với mọi j => tổng q_k=0 <= eps => round bị bỏ.
    rejected_score_matrix = torch.full((N, N), -0.3)
    weights_bad, diag_bad = tc_lora.compute_trust_weights(rejected_score_matrix)
    print("round_accepted:", diag_bad["round_accepted"], "| weights:", weights_bad)
    assert diag_bad["round_accepted"] is False
    assert weights_bad is None
    print("OK: TC-LoRA trả None đúng khi round bị bỏ")

    print("\n=== TẤT CẢ SMOKE TEST PASS ===")


if __name__ == "__main__":
    main()
