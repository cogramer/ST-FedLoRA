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
from typing import Any, Dict, List

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
) -> Dict[str, Any]:
    """
    Dựng toàn bộ dữ liệu cục bộ cho 5 Client: partition Train global thành
    5 phần (mục 4.4), tách D_i^fit/D_i^trust mỗi Client (10%, mục 4.4), rồi
    với riêng Client Byzantine, dựng D_clean/D_poison từ D_i^fit của nó
    (mục 5.2, ρ_poison=0.20, nhãn mục tiêu theo dataset).

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
) -> Dict[str, Any]:
    """
    1 lần chạy đầy đủ (1 dataset, 1 partition_mode, 1 rank, 1 training_seed):
    Phase 1 Warm-up (chung) rồi Phase 2 x 5 nhánh độc lập (mục 7.1).
    """
    seed_partition = 500_000 + training_seed
    seed_fit_trust = 600_000 + training_seed
    seed_poison_select = 700_000 + training_seed

    inputs = build_experiment_inputs(
        dataset_name, data_root, partition_mode,
        seed_partition=seed_partition, seed_fit_trust=seed_fit_trust,
        seed_poison_select=seed_poison_select, byzantine_client_index=byzantine_client_index,
    )

    # ---- Phase 1: Warm-up Head (mục 4.3.1). Không có Byzantine/Trigger ở
    # giai đoạn này -- Client sẽ-là-Byzantine dùng D_clean (chưa Trigger). ----
    warmup_cfg = LoRAModelConfig(num_labels=inputs["num_labels"])
    warmup_model = build_warmup_model(warmup_cfg)
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
        finalize_cfg=CCMaliciousLoRASGDConfig(
            cosine=CosineProjectionConfig(tau=0.85),
            lambda_poison=1.0,
            gaussian_sigma=0.05,
            gaussian_seed=999,
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
    out_path = os.path.join(results_dir, f"{dataset_name}_{partition_mode}_r{lora_rank}_seed{training_seed}.json")
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
    args = parser.parse_args()

    run_full_experiment(
        dataset_name=args.dataset, data_root=args.data_root, partition_mode=args.partition,
        lora_rank=args.rank, training_seed=args.seed, device=args.device,
        num_rounds=args.num_rounds, results_dir=args.results_dir,
    )
