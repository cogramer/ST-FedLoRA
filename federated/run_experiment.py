"""
federated/run_experiment.py

Script orchestration cấp cao nhất (mục 7.1): build dữ liệu + partition,
Phase 1 Warm-up Head (1 lần, dùng CHUNG cho mọi nhánh), rồi Phase 2 chạy 5
nhánh ĐỘC LẬP (FedAvg, Krum, Multi-Krum, Trimmed Mean, TC-LoRA) từ CÙNG
checkpoint LoRA khởi tạo (A_0,B_0) và CÙNG partition/cấu hình tấn công, lưu
kết quả vào results/.

Đây là bản khung đầu tiên, CHƯA gắn configs/*.yaml (việc còn lại kế tiếp
theo tiến trình dự án) -- tham số mặc định lấy theo đúng bảng mục 8.

NẾU tên hàm trong data/loaders.py, data/partition.py, data/poison.py,
models/modernbert_lora.py ở repo thực tế khác với import bên dưới (ví dụ
load_dbpedia_csv có thể tên khác), chỉ cần sửa đúng chỗ import -- cấu trúc
gọi hàm (tham số, thứ tự) giữ nguyên theo mô tả trong
/areas/st-fedlora-progress.md.

byzantine_cfg.finalize_cfg (CosineProjectionConfig/CCMaliciousLoRASGDConfig
thật trong attack/cc_malicious_lora_sgd.py, gồm tau, eps_ref, eps_q, sigma,
gaussian_seed) PHẢI được gán trước khi chạy thật -- để mặc định None ở đây
sẽ lỗi ngay khi finalize_malicious_update được gọi, đó là CHỦ Ý (fail-fast
thay vì âm thầm bỏ qua cosine projection).
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from attack.cc_malicious_lora_sgd import CCMaliciousLoRASGDConfig, CosineProjectionConfig, MaliciousSGD
from data.loaders import load_dataset_bundle
from data.partition import partition_dataset, split_fit_trust
from data.poison import PoisonConfig, build_poisoned_fit_set, get_target_label_id
from federated.client import BenignClientConfig, ByzantineClientConfig
from federated.dataset import make_dataloader
from federated.server import (
    FederatedLoRAConfig,
    FederatedLoRAExperiment,
    WarmupConfig,
    run_head_warmup,
)
from models.modernbert_lora import (
    LoRAModelConfig,
    build_federated_model,
    build_warmup_model,
    build_tokenizer,
    get_lora_state_dict,
)

SUPPORTED_MECHANISMS = ("fedavg", "krum", "multi_krum", "trimmed_mean", "tc_lora")


def subsample_dataframe(df, max_samples: Optional[int], seed: int, label_col: str = "label"):
    """
    Lấy ngẫu nhiên (CÓ PHÂN TẦNG theo nhãn, giữ TỈ LỆ lớp gốc) 1 subset tối
    đa `max_samples` dòng từ `df`, dùng RIÊNG cho mục đích DEBUG/chạy thử
    (không phải 1 phần công thức thí nghiệm trong đề cương).

    max_samples=None hoặc >= len(df) -> trả về nguyên `df` (không cắt).

    Phân tầng theo lớp (thay vì random.sample thuần) để subset nhỏ vẫn có
    đủ mọi nhãn với tỉ lệ gần giống bản gốc -- quan trọng vì
    data/poison.py.build_poisoned_fit_set CẦN đủ số mẫu KHÔNG mang nhãn mục
    tiêu để chọn ra tập D_poison theo đúng poison_ratio; nếu subset quá nhỏ
    hoặc lệch quá nhiều, hàm đó sẽ báo lỗi rõ ràng (fail-fast) thay vì âm
    thầm tạo ra tập poison không đúng tỉ lệ -- đây là HÀNH VI CHỦ Ý, không
    phải lỗi cần che giấu.

    seed: seed RIÊNG cho việc lấy mẫu subset (KHÔNG dùng chung với bất kỳ
    seed thí nghiệm nào khác -- partition, fit/trust, poison, LoRA init...),
    đúng quy ước "mỗi nguồn ngẫu nhiên 1 seed độc lập" của dự án.
    """
    if max_samples is None or max_samples >= len(df):
        return df

    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    labels = df[label_col].to_numpy()
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(df)

    selected_indices: List[int] = []
    for label, count in zip(unique_labels, counts):
        # Phân bổ proportionally theo tỉ lệ lớp gốc, tối thiểu 1 mẫu/lớp
        # (nếu lớp đó có ít nhất 1 mẫu) để không "biến mất" nhãn hiếm khi
        # subset nhỏ.
        n_take = max(1, round(max_samples * count / total))
        n_take = min(n_take, count)
        class_indices = np.where(labels == label)[0]
        chosen = rng.choice(class_indices, size=n_take, replace=False)
        selected_indices.extend(chosen.tolist())

    selected_indices = np.array(selected_indices)
    # Có thể lệch vài mẫu so với max_samples do làm tròn/min 1 mẫu/lớp --
    # nếu dư, cắt bớt ngẫu nhiên (vẫn giữ seed, tái lập được); nếu thiếu
    # (hiếm, do min 1 mẫu/lớp đẩy tổng lên), giữ nguyên không cố nhồi thêm.
    if len(selected_indices) > max_samples:
        selected_indices = rng.choice(selected_indices, size=max_samples, replace=False)

    rng.shuffle(selected_indices)
    return df.iloc[selected_indices].reset_index(drop=True)


def build_experiment_inputs(
    dataset_name: str,
    data_root: str,
    partition_mode: str,
    seed_partition: int,
    seed_fit_trust: int,
    seed_poison_select: int,
    byzantine_client_index: int = 4,
    trust_ratio: float = 0.10,
    poison_ratio: float = 0.20,
    trigger_text: str = " [TRIGGER]",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    subset_seed: int = 2026,
) -> Dict[str, Any]:
    """
    Dựng toàn bộ dữ liệu cục bộ cho 5 Client: partition Train global thành
    5 phần (mục 4.4), tách D_i^fit/D_i^trust mỗi Client (10%, mục 4.4), rồi
    với riêng Client Byzantine, dựng D_clean/D_poison từ D_i^fit của nó
    (mục 5.2, ρ_poison=0.20, nhãn mục tiêu theo dataset).

    max_train_samples/max_val_samples/max_test_samples: CHỈ để debug/chạy
        thử nhanh (vd kiểm tra pipeline không bị treo) -- lấy subset phân
        tầng theo nhãn (xem subsample_dataframe) từ bundle.train_df/val_df/
        test_df TRƯỚC KHI partition cho 5 Client. None (mặc định) -> dùng
        TOÀN BỘ dataset như cũ, không ảnh hưởng tới kết quả thí nghiệm
        chính thức. subset_seed: seed RIÊNG cho việc lấy subset, mặc định
        2026 để không trùng bất kỳ seed thí nghiệm nào khác trong dự án.

        LƯU Ý: nếu max_train_samples quá nhỏ, build_poisoned_fit_set (mục
        5.2, ρ_poison=0.20) có thể không tìm đủ mẫu KHÔNG mang nhãn mục
        tiêu trong D_i^fit của Client Byzantine để tạo D_poison đúng tỉ lệ
        -> hàm đó sẽ raise lỗi rõ ràng. Đây là HÀNH VI ĐÚNG (fail-fast),
        KHÔNG cố né bằng cách tự ý hạ poison_ratio hay nới điều kiện --
        nếu gặp lỗi này, hãy tăng max_train_samples thay vì sửa poison.py.

    Trả về dict: client_fit_dfs (list[df|None], N=5 phần tử theo thứ tự
    Client 0..N-1, vị trí byzantine_client_index = None vì Byzantine dùng
    byzantine_df_clean/byzantine_df_poison riêng), trust_dfs (list[df], N
    phần tử CÙNG THỨ TỰ), num_samples (list[int], n_i^fit mỗi Client CÙNG
    THỨ TỰ), byzantine_df_clean, byzantine_df_poison, df_global_val,
    df_global_test, target_label_id, num_labels.
    """
    if dataset_name not in ("agnews_redis", "dbpedia"):
        raise ValueError(f"build_experiment_inputs: dataset_name '{dataset_name}' không hỗ trợ.")

    bundle = load_dataset_bundle(dataset_name, data_root)
    num_labels = bundle.num_classes

    if max_train_samples is not None:
        before = len(bundle.train_df)
        bundle.train_df = subsample_dataframe(bundle.train_df, max_train_samples, seed=subset_seed)
        print(f"[subset] train_df: {before} -> {len(bundle.train_df)} mẫu (max_train_samples={max_train_samples})")
    if max_val_samples is not None:
        before = len(bundle.val_df)
        bundle.val_df = subsample_dataframe(bundle.val_df, max_val_samples, seed=subset_seed + 1)
        print(f"[subset] val_df: {before} -> {len(bundle.val_df)} mẫu (max_val_samples={max_val_samples})")
    if max_test_samples is not None:
        before = len(bundle.test_df)
        bundle.test_df = subsample_dataframe(bundle.test_df, max_test_samples, seed=subset_seed + 2)
        print(f"[subset] test_df: {before} -> {len(bundle.test_df)} mẫu (max_test_samples={max_test_samples})")

    target_label_id = get_target_label_id(dataset_name)
    client_indices = partition_dataset(bundle.train_df["label"].to_numpy(), num_clients=5, scheme=partition_mode, seed=seed_partition)

    client_fit_dfs: List[Any] = []
    trust_dfs: List[Any] = []
    num_samples: List[int] = []
    byzantine_df_clean = None
    byzantine_df_poison = None

    fit_parts, trust_parts = split_fit_trust(client_indices, seed=seed_fit_trust, trust_ratio=trust_ratio)  
    for i in range(5):  
        df_fit = bundle.train_df.iloc[fit_parts[i]]  
        df_trust = bundle.train_df.iloc[trust_parts[i]]
        trust_dfs.append(df_trust)

        if i == byzantine_client_index:
            poison_cfg = PoisonConfig(
                target_label=target_label_id, trigger_text=trigger_text,
                poison_ratio=poison_ratio, seed=seed_poison_select,
            )
            poisoned = build_poisoned_fit_set(df_fit, poison_cfg)
            byzantine_df_clean = poisoned["D_clean"]
            byzantine_df_poison = poisoned["D_poison"]
            num_samples.append(len(byzantine_df_clean) + len(byzantine_df_poison))
            client_fit_dfs.append(None)
        else:
            client_fit_dfs.append(df_fit)
            num_samples.append(len(df_fit))

    return {
        "client_fit_dfs": client_fit_dfs,
        "trust_dfs": trust_dfs,
        "byzantine_df_clean": byzantine_df_clean,
        "byzantine_df_poison": byzantine_df_poison,
        "num_samples": num_samples,
        "df_global_val": bundle.val_df,
        "df_global_test": bundle.test_df,
        "target_label_id": target_label_id,
        "num_labels": num_labels,
    }


def simple_accuracy_eval(model, df, tokenizer, device, batch_size: int = 32, max_length: int = 256) -> float:
    """
    Eval TẠM THỜI chỉ phục vụ chọn checkpoint Head trong warm-up (Phase 1
    early-stopping). eval/ (mục 7.2, chưa viết) sẽ có bản chính thức dùng
    để BÁO CÁO kết quả cuối -- KHÔNG dùng hàm này cho ACC/ASR báo cáo.
    """
    import torch

    loader = make_dataloader(df, tokenizer, batch_size=batch_size, shuffle=False, max_length=max_length)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
    return correct / total if total > 0 else 0.0


def _strip_tensors_for_json(obj):
    """Chuyển tensor trong diagnostics thành list/float để json.dump không lỗi."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _strip_tensors_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_tensors_for_json(v) for v in obj]
    return obj


def run_full_experiment(
    dataset_name: str,
    data_root: str,
    partition_mode: str,
    lora_rank: int,
    training_seed: int,
    device: str = "cuda",
    byzantine_client_index: int = 4,
    num_rounds: int = 30,
    results_dir: str = "results",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    subset_seed: int = 2026,
    cosine_tau: float = 0.85,
    gaussian_sigma: float = 0.05,
    lambda_poison: float = 1.0,
    gaussian_seed: int = 999,
) -> Dict[str, Any]:
    """
    1 lần chạy đầy đủ (1 dataset, 1 partition_mode, 1 rank, 1 training_seed):
    Phase 1 Warm-up (chung) rồi Phase 2 x 5 nhánh độc lập (mục 7.1).

    max_train_samples/max_val_samples/max_test_samples: xem docstring
    build_experiment_inputs -- CHỈ dùng để chạy thử/debug nhanh trên 1
    subset, KHÔNG dùng cho kết quả thí nghiệm chính thức báo cáo trong
    khóa luận.

    cosine_tau/gaussian_sigma/lambda_poison: đây là 3 trong số các siêu
    tham số "khảo sát" của mục 7.5 (Sensitivity: τ, σ, λ_poison) -- đề
    cương KHÔNG cố định 1 giá trị duy nhất, mà yêu cầu khảo sát nhiều giá
    trị (τ∈{0.80,0.85,0.90}) rồi chọn bằng Validation. Trước đây các giá
    trị này bị hard-code thẳng trong hàm (tau=0.85, sigma=0.05,
    lambda_poison=1.0), KHÔNG thể đổi từ CLI -- nay expose ra tham số +
    CLI flag tương ứng (--cosine_tau/--gaussian_sigma/--lambda_poison) để
    chạy khảo sát nhiều giá trị mà không cần sửa code mỗi lần.
    """
    seed_partition = 500_000 + training_seed
    seed_fit_trust = 600_000 + training_seed
    seed_poison_select = 700_000 + training_seed

    inputs = build_experiment_inputs(
        dataset_name, data_root, partition_mode,
        seed_partition=seed_partition, seed_fit_trust=seed_fit_trust,
        seed_poison_select=seed_poison_select, byzantine_client_index=byzantine_client_index,
        max_train_samples=max_train_samples, max_val_samples=max_val_samples,
        max_test_samples=max_test_samples, subset_seed=subset_seed,
    )

    # ---- Phase 1: Warm-up Head (mục 4.3.1). Không có Byzantine/Trigger ở
    # giai đoạn này -- Client sẽ-là-Byzantine dùng D_clean (chưa Trigger). ----
    warmup_cfg = LoRAModelConfig(num_labels=inputs["num_labels"])
    warmup_model = build_warmup_model(warmup_cfg)

    # Kiểm tra các tham số được đóng băng và được huấn luyện
    trainable = [
        name for name, param in warmup_model.named_parameters()
        if param.requires_grad
    ]

    frozen = [
        name for name, param in warmup_model.named_parameters()
        if not param.requires_grad
    ]

    print("\nTrainable parameters:")
    for name in trainable:
        print(" ", name)

    print(f"\nTrainable tensors: {len(trainable)}")
    print(f"Frozen tensors: {len(frozen)}")

    tokenizer = build_tokenizer(warmup_cfg)
    warmup_client_fit_dfs = list(inputs["client_fit_dfs"])
    warmup_client_fit_dfs[byzantine_client_index] = inputs["byzantine_df_clean"]

    head_state, warmup_diag = run_head_warmup(
        warmup_model, tokenizer, warmup_client_fit_dfs, inputs["num_samples"],
        inputs["df_global_val"], WarmupConfig(), simple_accuracy_eval,
        round_seed_offset=100_000 + training_seed, device=device,
    )

    # ---- Phase 2: model với LoRA gắn sẵn + H* (Head đã warm-up) đóng băng. ----
    lora_cfg = LoRAModelConfig(num_labels=inputs["num_labels"], lora_rank=lora_rank, init_seed=800_000 + training_seed)  
    federated_model = build_federated_model(lora_cfg, head_state_dict=head_state)
    federated_model.load_state_dict(head_state, strict=False)
    initial_lora_state = get_lora_state_dict(federated_model)

    benign_cfg = BenignClientConfig()
    byzantine_cfg = ByzantineClientConfig(
        lambda_poison=lambda_poison,
        finalize_cfg=CCMaliciousLoRASGDConfig(
            cosine=CosineProjectionConfig(tau=cosine_tau),
            lambda_poison=lambda_poison,
            gaussian_sigma=gaussian_sigma,
            gaussian_seed=gaussian_seed,
        )
    )  # finalize_cfg PHẢI được gán trước khi chạy thật, xem docstring module

    benign_dfs_only = [df for i, df in enumerate(inputs["client_fit_dfs"]) if i != byzantine_client_index]

    results: Dict[str, Any] = {"warmup_diagnostics": warmup_diag, "branches": {}}

    for mechanism in SUPPORTED_MECHANISMS:
        malicious_optimizer = MaliciousSGD(
            [p for p in federated_model.parameters() if p.requires_grad],
            lr=byzantine_cfg.lr_malicious, beta=byzantine_cfg.beta, gamma=byzantine_cfg.gamma,
            r_min=byzantine_cfg.r_min, r_max=byzantine_cfg.r_max, eps_v=byzantine_cfg.eps_v,
        )
        agg_cfg = FederatedLoRAConfig(mechanism=mechanism, num_rounds=num_rounds, byzantine_index=byzantine_client_index)

        experiment = FederatedLoRAExperiment(
            model=federated_model,
            tokenizer=tokenizer,
            initial_lora_state=copy.deepcopy(initial_lora_state),
            benign_client_dfs=benign_dfs_only,
            byzantine_df_clean=inputs["byzantine_df_clean"],
            byzantine_df_poison=inputs["byzantine_df_poison"],
            trust_dfs=inputs["trust_dfs"],
            num_samples=inputs["num_samples"],
            benign_cfg=benign_cfg,
            byzantine_cfg=byzantine_cfg,
            malicious_optimizer=malicious_optimizer,
            agg_cfg=agg_cfg,
            device=device,
        )
        _final_lora_state, round_logs = experiment.run()
        results["branches"][mechanism] = {"round_logs": round_logs}

    os.makedirs(results_dir, exist_ok=True)
    # Tên file bao gồm cả tau/sigma/lambda_poison để các lần chạy sensitivity
    # sweep (mục 7.5) KHÔNG ghi đè lẫn nhau (trước đây hard-code 1 giá trị
    # duy nhất nên không cần phân biệt trong tên file; giờ có thể đổi từ
    # CLI nên phải phân biệt).
    out_path = os.path.join(
        results_dir,
        f"{dataset_name}_{partition_mode}_r{lora_rank}_seed{training_seed}"
        f"_tau{cosine_tau}_sigma{gaussian_sigma}_lam{lambda_poison}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_strip_tensors_for_json(results), f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chạy 1 lần thực nghiệm ST-FedLoRA đầy đủ (Phase 1 Warm-up + Phase 2 x 5 nhánh)."
    )
    parser.add_argument("--dataset", choices=["agnews_redis", "dbpedia"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--partition", choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--rank", type=int, default=8, choices=[2, 4, 8])
    parser.add_argument("--seed", type=int, default=42, choices=[42, 123, 2024])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_rounds", type=int, default=30)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="CHỈ để debug/chạy thử nhanh: giới hạn số mẫu Train global (phân tầng theo nhãn) "
             "trước khi partition cho 5 Client. Mặc định None = dùng toàn bộ dataset.",
    )
    parser.add_argument(
        "--max_val_samples", type=int, default=None,
        help="CHỈ để debug/chạy thử nhanh: giới hạn số mẫu Global Validation. Mặc định None = toàn bộ.",
    )
    parser.add_argument(
        "--max_test_samples", type=int, default=None,
        help="CHỈ để debug/chạy thử nhanh: giới hạn số mẫu Global Test. Mặc định None = toàn bộ.",
    )
    parser.add_argument(
        "--subset_seed", type=int, default=2026,
        help="Seed riêng cho việc lấy subset (không dùng chung với seed thí nghiệm khác).",
    )
    parser.add_argument(
        "--cosine_tau", type=float, default=0.85, choices=[0.80, 0.85, 0.90],
        help="Ngưỡng Cosine τ cho phép chiếu 5.5/5.6 (mục 7.5 khảo sát 0.80/0.85/0.90).",
    )
    parser.add_argument(
        "--gaussian_sigma", type=float, default=0.05,
        help="Độ lệch chuẩn nhiễu Gaussian σ (mục 5.6, 7.5 -- khảo sát, hiệu chỉnh theo norm cập nhật).",
    )
    parser.add_argument(
        "--lambda_poison", type=float, default=1.0,
        help="Hệ số λ_poison trong L_local=L_clean+λ_poison*L_poison (mục 5.2, 7.5 khảo sát).",
    )
    parser.add_argument(
        "--gaussian_seed", type=int, default=999,
        help="Seed riêng cho nhiễu Gaussian (mục 5.6), KHÔNG dùng chung với seed thí nghiệm khác.",
    )
    args = parser.parse_args()

    run_full_experiment(
        dataset_name=args.dataset, data_root=args.data_root, partition_mode=args.partition,
        lora_rank=args.rank, training_seed=args.seed, device=args.device,
        num_rounds=args.num_rounds, results_dir=args.results_dir,
        max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples, subset_seed=args.subset_seed,
        cosine_tau=args.cosine_tau, gaussian_sigma=args.gaussian_sigma,
        lambda_poison=args.lambda_poison, gaussian_seed=args.gaussian_seed,
    )
