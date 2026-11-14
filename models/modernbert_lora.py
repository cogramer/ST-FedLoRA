"""
models/modernbert_lora.py
Bọc ModernBERT-base (answerdotai/ModernBERT-base, HuggingFace Hub) bằng
PEFT-LoRA cho bài toán phân loại văn bản (AG News 4 lớp / DBpedia-L1 9 lớp).

Theo mục 4.2-4.3 đề cương:
- Backbone ModernBERT-base đóng băng hoàn toàn.
- Chỉ chèn adapter LoRA (A, B) vào các ma trận attention (query, value)
    -> đây là phần duy nhất được trao đổi giữa Client <-> Server.
- Lớp cuối là classification head không đóng băng -> vẫn train cùng LoRA,
  nhưng không nằm trong state trao đổi liên bang (xem get_lora_state_dict).

LoRA Aggregation Discordance:
    Server/Client chỉ trao đổi và tổng hợp A, B độc lập
    (vec(A_k), vec(B_k) riêng biệt), không gộp thành ΔW = B@A rồi mới
    tổng hợp. Đây là lỗ hổng "LoRA Aggregation Discordance" mà đề
    tài khai thác (B̄Ā ≠ Σ p_k(B_k A_k)) nên các hàm set/get state dict ở
    đây tách riêng lora_A và lora_B, không có hàm nào "merge" chúng
    trước khi gửi đi server.

Seed riêng cho init LoRA:
    PEFT mặc định khởi tạo ma trận A ~ Gaussian(0, std nhỏ), B = 0. Để đảm
    bảo seed khởi tạo LoRA độc lập với seed partition dữ liệu và seed nhiễu
    attacker, hàm `build_model()` nhận một `init_seed` riêng và chỉ set
    `torch.manual_seed` ngay trước lúc PEFT khởi tạo adapter, không set ở
    đâu khác trong toàn pipeline.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Cần `pip install -U transformers>=4.48` (ModernBERT yêu cầu bản mới)."
    ) from e

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError as e:  # pragma: no cover
    raise ImportError("Cần `pip install peft`.") from e


MODERNBERT_CHECKPOINT = "answerdotai/ModernBERT-base"

# Các module Linear bên trong ModernBERT mà LoRA sẽ chèn vào.
# ModernBERT dùng attention dạng "Wqkv" gộp chung (không tách query/key/value
# thành 3 Linear riêng như BERT cổ điển) ở một số bản HF, nhưng để an toàn và
# tương thích ngược, ta match theo TÊN MODULE CUỐI (suffix) thay vì full path
# -- PEFT hỗ trợ match theo suffix trong `target_modules`.
#   - "Wqkv"   : projection gộp Q,K,V (kiến trúc ModernBERT chuẩn trên HF)
#   - "Wo"     : output projection của attention
# Nếu checkpoint cụ thể tính dùng có tên khác thì in `model.named_modules()` ra
# để đối chiếu rồi chỉnh lại danh sách này.
DEFAULT_TARGET_MODULES = ["Wqkv", "Wo"]


@dataclass
class LoRAModelConfig:
    """Cấu hình để dựng 1 model ModernBERT+LoRA, khớp 1-1 với configs/*.yaml."""
    num_labels: int
    lora_rank: int = 8                       # r trong đề cương (thử 4, 8, 16)
    lora_alpha: int = 16                     # thường = 2*r theo khuyến nghị PEFT
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    max_length: int = 128
    checkpoint: str = MODERNBERT_CHECKPOINT
    init_seed: int = 0                       # seed RIÊNG cho khởi tạo ma trận A
    cache_dir: Optional[str] = None          # vd: "/content/drive/MyDrive/hf_cache"



# Tokenizer dùng chung không có trọng số train nên không cần qua LoRA

def build_tokenizer(cfg: LoRAModelConfig):
    return AutoTokenizer.from_pretrained(cfg.checkpoint, cache_dir=cfg.cache_dir)


def build_model(cfg: LoRAModelConfig) -> PeftModel:
    """
    Trả về 1 PeftModel sẵn sàng train:
        - Toàn bộ trọng số gốc ModernBERT: requires_grad = False (đóng băng)
        - Adapter LoRA (lora_A, lora_B): requires_grad = True
        - Classification head (score/classifier): requires_grad = True
          (head luôn được PEFT để "modules_to_save" mặc định khi dùng
           AutoModelForSequenceClassification nhưng ta vẫn khai báo tường minh
           cho chắc ăn)
    """
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.checkpoint,
        num_labels=cfg.num_labels,
        cache_dir=cfg.cache_dir,
    )

    for param in base_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="SEQ_CLS",
        # modules_to_save: các module không áp LoRA nhưng vẫn train toàn phần
        # (ở đây là classification head) head này không được trao đổi khi làm
        # Federated Learning (xem lưu ý get_lora_state_dict bên dưới).
        modules_to_save=["classifier", "score"],
    )

    # Seed riêng cho khởi tạo ma trận LoRA A (Gaussian)
    # chỉ set ngay trước get_peft_model() để không ảnh hưởng seed partition
    # dữ liệu hay seed nhiễu attacker ở nơi khác trong pipeline.
    torch.manual_seed(cfg.init_seed)
    model = get_peft_model(base_model, lora_config)

    return model


def count_trainable_parameters(model: nn.Module) -> dict[str, int]:
    """In ra số tham số trainable so với tổng số tham số để sanity-check là
    LoRA thật sự chỉ chiếm phần nhỏ (~0.1-1%) tổng số tham số ModernBERT."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
    }


def get_lora_state_dict(model: PeftModel) -> dict[str, torch.Tensor]:
    """
    Trả về dict {tên_param: tensor} chỉ chứa lora_A.* và lora_B.* (đã clone và
    chuyển CPU để an toàn khi serialize hay gửi qua mạng hoặc lưu checkpoint).

    Không bao gồm classifier/score (head) vì nó không nằm trong phạm vi 
    tổng hợp liên bang. Mỗi client có thể giữ head riêng hoặc
    head được xử lý ở tầng khác tuỳ thiết kế thực nghiệm cụ thể về sau.
    """
    state = model.state_dict()
    lora_state = {
        name: tensor.detach().clone().cpu()
        for name, tensor in state.items()
        if "lora_A" in name or "lora_B" in name
    }
    if not lora_state:
        raise RuntimeError(
            "Không tìm thấy tham số lora_A/lora_B nào trong model. "
            "Kiểm tra lại target_modules có khớp với kiến trúc ModernBERT không."
        )
    return lora_state


def set_lora_state_dict(model: PeftModel, lora_state: dict[str, torch.Tensor]) -> None:
    """
    Nạp ngược 1 dict {lora_A.*, lora_B.*} vào model. Dùng khi Client nhận
    (A_t, B_t) mới từ Server ở đầu mỗi round. `strict=False` vì dict này chỉ
    chứa 1 phần state vì không có backbone và không có head.
    """
    full_state = model.state_dict()
    missing = set(lora_state.keys()) - set(full_state.keys())
    if missing:
        raise KeyError(f"Các khoá sau không khớp với model hiện tại: {missing}")

    device = next(model.parameters()).device
    full_state.update({k: v.to(device) for k, v in lora_state.items()})
    model.load_state_dict(full_state, strict=False)


def clone_lora_state_dict(lora_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Deep-copy 1 state dict LoRA. Dùng khi cần giữ bản gốc trước khi
    Client chỉnh sửa cục bộ (vd. tính update ΔA = A_local - A_global)."""
    return {k: v.clone() for k, v in lora_state.items()}


def lora_state_dict_to_flat_vector(lora_state: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Làm phẳng toàn bộ state (A và B gộp theo thứ tự key cố định) thành 1
    vector 1-D. Dùng cho các cơ chế phòng thủ cần so sánh khoảng cách giữa
    các client (Krum, Trimmed Mean, FedGraph). Các module đó sẽ tự quyết
    định có tách A/B ra so sánh riêng hay gộp chung, hàm này chỉ cung cấp
    tiện ích làm phẳng với thứ tự khoá được sorted nhất quán để mọi client so
    sánh đúng vị trí với nhau.
    """
    sorted_keys = sorted(lora_state.keys())
    return torch.cat([lora_state[k].flatten() for k in sorted_keys])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check build ModernBERT+LoRA")
    parser.add_argument("--num_labels", type=int, default=4, help="4=AG News, 9=DBpedia")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = LoRAModelConfig(
        num_labels=args.num_labels,
        lora_rank=args.lora_rank,
        cache_dir=args.cache_dir,
    )

    print(f"Đang tải {cfg.checkpoint} (lần đầu sẽ tải về, các lần sau dùng cache)...")
    model = build_model(cfg)
    tokenizer = build_tokenizer(cfg)

    stats = count_trainable_parameters(model)
    print(f"\nTham số trainable: {stats['trainable']:,} / {stats['total']:,} "
          f"({stats['trainable_pct']:.3f}%)")

    print("\n--- Tên module có LoRA (vài dòng đầu) ---")
    for name, _ in model.named_modules():
        if "lora_A" in name or "lora_B" in name:
            print(" ", name)

    lora_state = get_lora_state_dict(model)
    print(f"\nSố tensor trong lora_state_dict: {len(lora_state)}")
    flat = lora_state_dict_to_flat_vector(lora_state)
    print(f"Kích thước vector LoRA làm phẳng: {flat.shape[0]:,} phần tử")

    # Test forward pass nhanh với câu giả
    sample = tokenizer(
        ["This is a test sentence for ModernBERT."],
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_length,
        padding="max_length",
    )
    model.eval()
    with torch.no_grad():
        out = model(**sample)
    print(f"\nForward pass OK -- logits shape: {out.logits.shape}")

    # Test reproducibility của init_seed
    cfg2 = LoRAModelConfig(num_labels=args.num_labels, lora_rank=args.lora_rank,
                            init_seed=cfg.init_seed, cache_dir=args.cache_dir)
    model2 = build_model(cfg2)
    state1 = get_lora_state_dict(model)
    state2 = get_lora_state_dict(model2)
    same_init = all(torch.equal(state1[k], state2[k]) for k in state1)
    print(f"Cùng init_seed -> cùng khởi tạo LoRA: {same_init}")