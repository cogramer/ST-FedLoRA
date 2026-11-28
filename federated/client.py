"""
federated/client.py

Logic huấn luyện cục bộ cho Client lành tính và Client Byzantine (mục 4.4,
5.1-5.6, 6.2 bước 2 đề cương). Server (federated/server.py) chỉ gọi vào
đây, KHÔNG tự cài lại vòng lặp training.

QUY ƯỚC CHUNG:
    - Encoder và Classification Head (H*) LUÔN đóng băng trong giai đoạn
      Federated LoRA chính (mục 4.3.1); chỉ (A, B) được huấn luyện. Model
      truyền vào mọi hàm ở đây PHẢI đã ở chế độ
      models.modernbert_lora.build_federated_model() (LoRA gắn sẵn,
      Encoder+Head requires_grad=False).
    - Mọi hàm nhận `global_lora_state` = (A_t, B_t) dưới dạng state dict
      LoRA ĐẦY ĐỦ (không phải delta) và trả về state dict LoRA ĐẦY ĐỦ sau
      huấn luyện cục bộ (A_{i,t+1}^local, B_{i,t+1}^local) -- Server tự trừ
      global_lora_state để ra delta (mục 3.4) bằng
      attack.cc_malicious_lora_sgd.compute_delta_state_dict.
    - Seed cục bộ: shuffle minibatch, chọn mẫu poison, nhiễu Gaussian, init
      optimizer PHẢI dùng seed RIÊNG truyền vào, KHÔNG set seed global.

NẾU tên hàm/tham số trong attack/cc_malicious_lora_sgd.py hoặc
models/modernbert_lora.py ở repo thực tế khác với bên dưới, chỉ cần sửa
đúng chỗ import/gọi hàm tương ứng (xem ghi chú trong
federated/aggregation/base.py cho cùng kiểu vấn đề).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch.optim import AdamW

from attack.cc_malicious_lora_sgd import (
    MaliciousSGD,
    CCMaliciousLoRASGDConfig,
    combine_local_loss,
    finalize_malicious_update,
    flatten_delta_state_dict,
)
from federated.dataset import make_dataloader
from models.modernbert_lora import get_lora_state_dict, set_lora_state_dict

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm luôn nên có trong môi trường Colab
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        """Fallback không phụ thuộc tqdm: trả về chính iterable, bỏ qua args."""
        return iterable if iterable is not None else range(0)

StateDict = "OrderedDict[str, torch.Tensor]"


@dataclass
class BenignClientConfig:
    """mục 4.4 / bảng mục 8: AdamW, lr=3e-4, weight_decay=0.01, E=3, batch=32."""

    lr: float = 3e-4
    weight_decay: float = 0.01
    local_epochs: int = 3
    batch_size: int = 32
    max_length: int = 256


def train_benign_client(
    model,
    tokenizer,
    global_lora_state: Dict[str, torch.Tensor],
    df_fit,
    cfg: BenignClientConfig,
    shuffle_seed: int,
    device: str = "cuda",
    client_tag: str = "",
) -> Dict[str, torch.Tensor]:
    """
    1 Client lành tính, 1 round: nhận (A_t,B_t), huấn luyện E epoch AdamW
    trên D_i^fit, trả về state dict LoRA sau huấn luyện cục bộ.

    client_tag: chuỗi hiển thị trong log/tqdm (vd "round 3 | benign client
        1") chỉ để debug tiến trình, KHÔNG ảnh hưởng tới kết quả huấn luyện.
    """
    set_lora_state_dict(model, global_lora_state)
    model.to(device)
    model.train()

    loader = make_dataloader(
        df_fit, tokenizer, batch_size=cfg.batch_size, shuffle=True,
        seed=shuffle_seed, max_length=cfg.max_length,
    )
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    for epoch in range(cfg.local_epochs):
        running_loss = 0.0
        num_batches = 0
        pbar = tqdm(
            loader,
            desc=f"[benign {client_tag}] epoch {epoch + 1}/{cfg.local_epochs}",
            leave=False,
        )
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            outputs = model(**batch, labels=labels)
            outputs.loss.backward()
            optimizer.step()

            running_loss += outputs.loss.item()
            num_batches += 1
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

        avg_loss = running_loss / num_batches if num_batches > 0 else float("nan")
        print(f"[benign {client_tag}] epoch {epoch + 1}/{cfg.local_epochs} xong, loss trung bình = {avg_loss:.4f}")

    return get_lora_state_dict(model)


@dataclass
class ByzantineClientConfig:
    """
    Gộp tham số cho CC-MaliciousLoRA-SGD (mục 5.2-5.6, bảng mục 8).
    `finalize_cfg` được truyền NGUYÊN VẸN cho
    attack.cc_malicious_lora_sgd.finalize_malicious_update (tau, eps_ref,
    eps_q, sigma, gaussian_seed, ... -- xem đúng tên field thật của config
    class trong attack/cc_malicious_lora_sgd.py).
    """

    finalize_cfg: CCMaliciousLoRASGDConfig
    lambda_poison: float = 1.0
    local_epochs: int = 3
    batch_size: int = 32
    max_length: int = 256
    lr_malicious: float = 3e-4  # eta_m, mục 5.3 (không cố định trong đề cương, chọn ở cấu hình cục bộ)
    beta: float = 0.9
    gamma: float = 1.0
    r_min: float = 1.0
    r_max: float = 5.0
    eps_v: float = 1e-8


def _run_local_epochs(
    model, loader_clean, loader_poison, optimizer, cfg: ByzantineClientConfig, device: str,
    role_tag: str = "byzantine",
) -> None:
    """
    Chạy E epoch tối ưu L_local = L_clean + lambda_poison * L_poison (mục
    5.2), optimizer.step() theo MaliciousSGD (mục 5.3). Dùng chung cho cả
    shadow run (loader_poison rỗng -> loss_poison=0, tương đương chỉ tối ưu
    L_clean) và main malicious run (loader_poison đầy đủ).

    loader_poison rỗng (list()) khi gọi cho shadow run (mục 5.4: shadow chỉ
    train trên D_clean). Khi khác rỗng, loader ngắn hơn được lặp vòng qua
    itertools.cycle để mọi step của loader_clean đều ghép được 1 batch poison.

    role_tag: chuỗi hiển thị trong log/tqdm (vd "round 3 | shadow" hoặc
        "round 3 | malicious") để phân biệt shadow run / main malicious run
        khi theo dõi tiến trình, KHÔNG ảnh hưởng tới kết quả huấn luyện.
    """
    has_poison = len(loader_poison) > 0
    for epoch in range(cfg.local_epochs):
        running_loss_clean = 0.0
        running_loss_poison = 0.0
        num_batches = 0
        poison_iter = itertools.cycle(loader_poison) if has_poison else None
        pbar = tqdm(
            loader_clean,
            desc=f"[{role_tag}] epoch {epoch + 1}/{cfg.local_epochs}",
            leave=False,
        )
        for batch_clean in pbar:
            batch_clean = {k: v.to(device) for k, v in batch_clean.items()}
            labels_clean = batch_clean.pop("labels")

            optimizer.zero_grad()
            out_clean = model(**batch_clean, labels=labels_clean)
            loss_clean = out_clean.loss

            if poison_iter is not None:
                batch_poison = next(poison_iter)
                batch_poison = {k: v.to(device) for k, v in batch_poison.items()}
                labels_poison = batch_poison.pop("labels")
                out_poison = model(**batch_poison, labels=labels_poison)
                loss_poison = out_poison.loss
            else:
                loss_poison = torch.zeros((), device=loss_clean.device)

            loss_total = combine_local_loss(loss_clean, loss_poison, cfg.lambda_poison)
            loss_total.backward()
            optimizer.step()

            running_loss_clean += loss_clean.item()
            running_loss_poison += loss_poison.item()
            num_batches += 1
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    loss_clean=f"{running_loss_clean / num_batches:.4f}",
                    loss_poison=f"{running_loss_poison / num_batches:.4f}",
                )

        avg_clean = running_loss_clean / num_batches if num_batches > 0 else float("nan")
        avg_poison = running_loss_poison / num_batches if num_batches > 0 else float("nan")
        print(
            f"[{role_tag}] epoch {epoch + 1}/{cfg.local_epochs} xong, "
            f"loss_clean trung bình = {avg_clean:.4f}"
            + (f", loss_poison trung bình = {avg_poison:.4f}" if has_poison else "")
        )


def train_byzantine_client(
    model,
    tokenizer,
    global_lora_state: Dict[str, torch.Tensor],
    df_clean,
    df_poison,
    cfg: ByzantineClientConfig,
    malicious_optimizer: MaliciousSGD,
    shuffle_seed_clean: int,
    shuffle_seed_poison: int,
    shuffle_seed_shadow: int,
    device: str = "cuda",
    round_tag: str = "",
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    1 Client Byzantine, 1 round (mục 5.1-5.6):

    1. Shadow run (5.4): từ (A_t,B_t), train CHỈ trên D_clean bằng 1
       MaliciousSGD MỚI (velocity=0, KHÔNG dùng chung state với
       malicious_optimizer) -> (A_ref,B_ref). Shadow run là 1 lượt thử
       nghiệm riêng mỗi round, không phải tiến trình liên tục của attacker.
    2. Main run (5.2+5.3): từ (A_t,B_t) (reset lại sau shadow run), train
       trên L_local=L_clean+lambda_poison*L_poison bằng `malicious_optimizer`
       (instance DUY NHẤT được caller giữ xuyên suốt nhiều round, vận tốc
       không reset giữa các round, chỉ trọng số LoRA được đồng bộ lại từ
       Server mỗi round qua set_lora_state_dict) -> (A_mal,B_mal) RAW.
    3. finalize_malicious_update(state_t, state_mal, state_ref, finalize_cfg)
       áp dụng cosine projection (5.5) + Gaussian perturbation & reproject
       (5.6) -> final_state, diagnostics.

    Trả về (final_state, diagnostics). Server tự trừ global_lora_state để ra
    delta Byzantine như MỌI Client khác (không có đường xử lý riêng ở tầng
    Server/aggregator -- đúng threat model: Server không biết Client nào
    Byzantine, mục 5.1).
    """
    print(f"[byzantine {round_tag}] bắt đầu: shadow run (D_clean) rồi main malicious run (D_clean+D_poison)")

    # ---- 1. Shadow clean reference run (5.4) ----
    set_lora_state_dict(model, global_lora_state)
    model.to(device)
    model.train()

    shadow_loader = make_dataloader(
        df_clean, tokenizer, batch_size=cfg.batch_size, shuffle=True,
        seed=shuffle_seed_shadow, max_length=cfg.max_length,
    )
    shadow_optimizer = MaliciousSGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_malicious, beta=cfg.beta, gamma=cfg.gamma,
        r_min=cfg.r_min, r_max=cfg.r_max, eps_v=cfg.eps_v,
    )
    _run_local_epochs(
        model, shadow_loader, loader_poison=[], optimizer=shadow_optimizer, cfg=cfg, device=device,
        role_tag=f"byzantine {round_tag} | shadow",
    )
    state_ref = get_lora_state_dict(model)
    print(f"[byzantine {round_tag}] shadow run xong -> có state_ref")

    # ---- 2. Main malicious run (5.2+5.3), optimizer state liên tục qua round ----
    set_lora_state_dict(model, global_lora_state)  # reset lại về (A_t,B_t) trước main run
    model.train()

    clean_loader = make_dataloader(
        df_clean, tokenizer, batch_size=cfg.batch_size, shuffle=True,
        seed=shuffle_seed_clean, max_length=cfg.max_length,
    )
    poison_loader = (
        make_dataloader(
            df_poison, tokenizer, batch_size=cfg.batch_size, shuffle=True,
            seed=shuffle_seed_poison, max_length=cfg.max_length,
        )
        if len(df_poison) > 0 else []
    )

    _run_local_epochs(
        model, clean_loader, poison_loader, malicious_optimizer, cfg, device,
        role_tag=f"byzantine {round_tag} | malicious",
    )
    state_mal = get_lora_state_dict(model)
    print(f"[byzantine {round_tag}] main malicious run xong -> có state_mal, áp dụng cosine projection + Gaussian...")

    # ---- 3. Cosine projection + Gaussian perturbation & reproject (5.5-5.6) ----
    final_state, diagnostics = finalize_malicious_update(
        global_lora_state, state_mal, state_ref, cfg.finalize_cfg
    )
    print(f"[byzantine {round_tag}] hoàn tất finalize_malicious_update")
    return final_state, diagnostics


def compute_validation_descent_direction(
    model,
    tokenizer,
    global_lora_state: Dict[str, torch.Tensor],
    df_trust,
    batch_size: int,
    max_length: int = 256,
    device: str = "cuda",
) -> torch.Tensor:
    """
    TC-LoRA mục 6.2 bước 2: d_i^val = -grad_{(A,B)} L_i^val(A_t,B_t;H*),
    tính TẠI TRẠNG THÁI GLOBAL ĐẦU VÒNG (A_t,B_t), KHÔNG cập nhật tham số.

    Dùng FULL-BATCH gradient trên toàn bộ D_i^trust (không chia nhiều
    optimizer step): đây là 1 hướng giảm loss duy nhất mỗi round, không
    phải 1 quá trình huấn luyện nhiều bước -- quyết định thiết kế vì đề
    cương không chỉ định full-batch hay nhiều mini-batch rồi cộng dồn;
    full-batch khớp đúng nghĩa công thức g_i^val=∇L_i^val(A_t,B_t;H*) (1
    gradient duy nhất, không phải trung bình nhiều gradient step).

    Trả về vector đã flatten (CÙNG quy ước Concat(vec(ΔA),vec(ΔB)) với
    candidate deltas) để dùng trực tiếp trong
    federated.aggregation.tc_lora.consistency_matrix.
    """
    set_lora_state_dict(model, global_lora_state)
    model.to(device)
    model.eval()

    lora_param_names = set(global_lora_state.keys())
    for name, p in model.named_parameters():
        p.requires_grad_(name in lora_param_names)

    loader = make_dataloader(df_trust, tokenizer, batch_size=batch_size, shuffle=False, max_length=max_length)

    model.zero_grad()
    total_loss = None
    total_count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        outputs = model(**batch, labels=labels)
        # Cộng dồn loss có trọng số theo số mẫu để tương đương loss trung
        # bình trên TOÀN BỘ D_i^trust (tránh lệch khi batch cuối nhỏ hơn).
        batch_count = labels.shape[0]
        batch_loss_sum = outputs.loss * batch_count
        total_loss = batch_loss_sum if total_loss is None else total_loss + batch_loss_sum
        total_count += batch_count

    if total_loss is None or total_count == 0:
        raise ValueError(
            "Không thể tính được hướng validation descent: "
            "tập dữ liệu trust trống rỗng."
        )

    val_loss = total_loss / total_count
    val_loss.backward()

    # .cpu() ở đây BẮT BUỘC để khớp quy ước xuyên suốt dự án: mọi state dict
    # LoRA/delta "gửi về Server" đều ở CPU (xem models/modernbert_lora.py::
    # get_lora_state_dict -- cùng lý do "an toàn khi serialize/gửi qua
    # mạng"). client_deltas (dùng làm `candidates` trong
    # federated/aggregation/tc_lora.py::consistency_matrix) được tính từ
    # get_lora_state_dict nên LUÔN LÀ CPU; nếu d_i^val ở đây để nguyên trên
    # `device` (thường là CUDA), `references` (xây từ d_i^val) và
    # `candidates` sẽ lệch device, khiến
    # `cosine_similarity_matrix`/`consistency_matrix` lỗi RuntimeError
    # "Expected all tensors to be on the same device" (lỗi thực tế gặp phải
    # khi chạy thật trên Colab, chỉ lộ ra ở nhánh TC-LoRA vì đây là cơ chế
    # DUY NHẤT ghép d_i^val với client_deltas).
    grad_state = {
        name: (p.grad.detach().clone().cpu() if p.grad is not None else torch.zeros_like(p).cpu())
        for name, p in model.named_parameters()
        if name in lora_param_names
    }
    descent_direction_state = {name: -g for name, g in grad_state.items()}
    return flatten_delta_state_dict(descent_direction_state)