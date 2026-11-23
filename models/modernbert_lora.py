"""
models/modernbert_lora.py
Bọc ModernBERT-base (answerdotai/ModernBERT-base, HuggingFace Hub) bằng
PEFT-LoRA cho bài toán phân loại văn bản (AG News Redis 4 lớp / DBpedia-L1 9 lớp).

Theo mục 4.2-4.3-4.3.1-4.3.2 đề cương, quy trình chia làm HAI giai đoạn tách
biệt, mỗi giai đoạn có một "chế độ" model khác nhau:

GIAI ĐOẠN 1 - Federated Classification Head Warm-up (mục 4.3.1):
    - Encoder đóng băng hoàn toàn.
    - KHÔNG gắn LoRA.
    - CHỈ Classification Head (các tham số thuộc `head` và `classifier` của
      ModernBertForSequenceClassification) được huấn luyện, tổng hợp bằng
      FedAvg trên 5 Client, không có Byzantine Client / mẫu Trigger.
    - Dùng `build_warmup_model()`.
    - Kết thúc giai đoạn này, chọn 1 checkpoint Head H* bằng Global Validation.

GIAI ĐOẠN 2 - Federated LoRA chính (mục 4.3, 5, 6):
    - Encoder vẫn đóng băng.
    - Head được NẠP LẠI từ H* (checkpoint chọn ở giai đoạn 1) và ĐÓNG BĂNG
      xuyên suốt giai đoạn này (đề cương: "Server phát tán cùng một
      Classification Head H* đến năm Client và đóng băng Head trong toàn bộ
      giai đoạn Federated LoRA chính").
    - CHỈ ma trận LoRA A, B được huấn luyện và trao đổi với Server.
    - Dùng `build_federated_model()`.

Quan trọng: hai giai đoạn này dùng hai hàm dựng model khác nhau, KHÔNG dùng
chung một `build_model()` gắn LoRA + mở khoá Head đồng thời như bản trước đây
của file này (bản đó không còn khớp đề cương vì Head sẽ bị train cùng lúc với
LoRA ở giai đoạn chính, trái với mục 4.3.1).

Vị trí gắn LoRA (mục 4.3.2):
    Trong ModernBERT, Query/Key/Value được hợp nhất trong MỘT lớp tuyến tính
    tên `Wqkv`. Cấu hình PEFT CHUẨN của nghiên cứu CHỈ gắn LoRA vào `Wqkv`:
        target_modules = ["Wqkv"]
    Đề cương ghi rõ đây KHÔNG tương đương với việc chỉ gắn riêng Query và
    Value (sẽ cần cơ chế cắt lát/custom adapter khác, không thuộc phạm vi
    khảo sát). Vì vậy KHÔNG thêm "Wo" (output projection) hay
    ["query", "value"] vào danh sách này.

Rank và scale (mục 4.3.2, bảng mục 8):
    r ∈ {2, 4, 8}; cố định tỉ lệ alpha/r = 2 -> alpha tương ứng là 4, 8, 16.
    `LoRAModelConfig.lora_alpha` mặc định None và được tự tính = 2*r trong
    `__post_init__`, để không ai vô tình đặt alpha cố định không đổi theo r.

LoRA Aggregation Discordance (mục 3.4):
    Server/Client chỉ trao đổi và tổng hợp A, B độc lập (factor-wise:
    vec(A_k), vec(B_k) riêng biệt), không gộp thành ΔW = B@A rồi mới tổng
    hợp. Các hàm get/set state dict ở đây tách riêng lora_A và lora_B, không
    có hàm nào "merge" chúng trước khi gửi đi server.

Seed riêng cho init LoRA:
    PEFT mặc định khởi tạo ma trận A ~ Gaussian(0, std nhỏ), B = 0. Để đảm
    bảo seed khởi tạo LoRA độc lập với seed partition dữ liệu và seed nhiễu
    attacker, `build_federated_model()` nhận một `init_seed` riêng (qua cfg)
    và chỉ set `torch.manual_seed` ngay trước lúc PEFT khởi tạo adapter,
    không set ở đâu khác trong toàn pipeline.
"""

from __future__ import annotations

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

# CHỈ "Wqkv" theo đúng mục 4.3.2 của đề cương (xem docstring đầu file).
# Chạy thử `model.named_modules()` trên đúng phiên bản Transformers đang dùng
# để xác nhận suffix "Wqkv" vẫn còn đúng (ghi chú kiểm tra trước của đề cương).
DEFAULT_TARGET_MODULES = ["Wqkv"]

# Tên các module con thực tế của ModernBertForSequenceClassification chứa
# Classification Head (prediction head "head": dense+activation+norm, và lớp
# tuyến tính cuối "classifier") -- xem mục 4.3: "Classification Head của
# ModernBERT áp dụng prediction head, dropout và lớp phân loại cuối".
# Lưu ý: bản trước của file này dùng modules_to_save=["classifier", "score"],
# nhưng "score" không phải tên module của kiến trúc ModernBERT (đó là tên
# dùng ở một số kiến trúc khác như Llama-for-classification) nên bị bỏ qua
# âm thầm; đồng thời "head" (phần dense+norm) đã bị thiếu hoàn toàn.
HEAD_MODULE_NAMES = ["head", "classifier"]


@dataclass
class LoRAModelConfig:
    """Cấu hình để dựng 1 model ModernBERT (+LoRA ở giai đoạn 2), khớp 1-1
    với configs/*.yaml."""
    num_labels: int
    lora_rank: int = 8                       # r trong đề cương (thử 2, 4, 8)
    lora_alpha: Optional[int] = None         # None -> tự tính = 2*lora_rank (alpha/r=2 cố định, mục 4.3.2)
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    max_length: int = 128
    checkpoint: str = MODERNBERT_CHECKPOINT
    init_seed: int = 0                       # seed RIÊNG cho khởi tạo ma trận A
    cache_dir: Optional[str] = None          # vd: "/content/drive/MyDrive/hf_cache"

    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = 2 * self.lora_rank


# Tokenizer dùng chung không có trọng số train nên không cần qua LoRA

def build_tokenizer(cfg: LoRAModelConfig):
    return AutoTokenizer.from_pretrained(cfg.checkpoint, cache_dir=cfg.cache_dir)


def _is_head_param(name: str) -> bool:
    """True nếu tên tham số thuộc Classification Head (head.* hoặc
    classifier.*), bất kể model đã bị PEFT bọc thêm tiền tố
    (vd. "base_model.model.classifier.weight") hay chưa."""
    parts = name.split(".")
    return any(p in HEAD_MODULE_NAMES for p in parts)


def _build_backbone(cfg: LoRAModelConfig) -> nn.Module:
    """Tải ModernBERT-base, đóng băng TOÀN BỘ tham số (kể cả Head). Dùng
    chung cho cả build_warmup_model và build_federated_model; mỗi hàm sẽ tự
    mở khoá phần tham số phù hợp với giai đoạn của mình."""
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.checkpoint,
        num_labels=cfg.num_labels,
        cache_dir=cfg.cache_dir,
    )
    for param in base_model.parameters():
        param.requires_grad = False
    return base_model


def build_warmup_model(cfg: LoRAModelConfig) -> nn.Module:
    """
    GIAI ĐOẠN 1 (mục 4.3.1): Federated Classification Head Warm-up.
    Trả về model KHÔNG có LoRA:
        - Encoder: requires_grad = False (đóng băng)
        - Head (head.* và classifier.*): requires_grad = True
    Không có Byzantine Client hay mẫu Trigger ở giai đoạn này; Server tổng
    hợp Head bằng FedAvg có trọng số theo số mẫu (xem federated/ cho phần
    vòng lặp huấn luyện/tổng hợp).
    """
    model = _build_backbone(cfg)
    head_found = False
    for name, param in model.named_parameters():
        if _is_head_param(name):
            param.requires_grad = True
            head_found = True
    if not head_found:
        raise RuntimeError(
            "Không tìm thấy tham số Head nào khớp HEAD_MODULE_NAMES "
            f"{HEAD_MODULE_NAMES}. Chạy model.named_parameters() để kiểm tra "
            "lại tên module Head thực tế của phiên bản Transformers đang dùng."
        )
    return model


def build_federated_model(
    cfg: LoRAModelConfig,
    head_state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> PeftModel:
    """
    GIAI ĐOẠN 2 (mục 4.3, 5, 6): Federated LoRA chính.

    - Encoder: đóng băng (như trước).
    - Head: nạp từ `head_state_dict` (checkpoint H* chọn ở giai đoạn 1, xem
      `get_head_state_dict`/warm-up), nạp TRƯỚC khi gọi `get_peft_model` nên
      cả hai nhánh mà PEFT tạo ra cho Head (original_module và
      modules_to_save.default) đều mang đúng trọng số H*. Sau khi tạo
      PeftModel, Head bị khoá lại tường minh (requires_grad=False) vì PEFT
      tự động mở khoá "classifier"/"score" cho mọi task_type="SEQ_CLS" bất
      kể cấu hình -- xem chú thích chi tiết ngay tại dòng khoá lại bên dưới.
      Nếu `head_state_dict=None` (vd. chạy smoke-test nhanh không qua
      warm-up), Head giữ nguyên trọng số khởi tạo ngẫu nhiên của checkpoint
      gốc nhưng vẫn bị đóng băng -- không dùng cấu hình này để báo cáo kết
      quả chính thức.
    - LoRA: chỉ gắn vào `cfg.target_modules` (mặc định ["Wqkv"]).
    """
    base_model = _build_backbone(cfg)

    if head_state_dict is not None:
        full_state = base_model.state_dict()
        missing = set(head_state_dict.keys()) - set(full_state.keys())
        if missing:
            raise KeyError(
                f"head_state_dict chứa khoá không khớp model hiện tại: {missing}"
            )
        device = next(base_model.parameters()).device
        full_state.update({k: v.to(device) for k, v in head_state_dict.items()})
        base_model.load_state_dict(full_state, strict=False)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="SEQ_CLS",
        # KHÔNG đưa Head vào modules_to_save: Head phải đóng băng ở giai
        # đoạn này (mục 4.3.1), không phải tham số trao đổi liên bang.
        modules_to_save=None,
    )

    # Seed riêng cho khởi tạo ma trận LoRA A (Gaussian); chỉ set ngay trước
    # get_peft_model() để không ảnh hưởng seed partition dữ liệu hay seed
    # nhiễu attacker ở nơi khác trong pipeline.
    torch.manual_seed(cfg.init_seed)
    model = get_peft_model(base_model, lora_config)

    # QUAN TRỌNG: với task_type="SEQ_CLS", PEFT's PeftModelForSequenceClassification
    # TỰ ĐỘNG đưa "classifier"/"score" vào modules_to_save ngay trong __init__
    # của nó, BẤT KỂ LoraConfig.modules_to_save được truyền là None hay không
    # (đây là hành vi cứng của thư viện PEFT, không có cờ cấu hình nào tắt
    # được). Hệ quả là Head bị PEFT tự bọc lại bằng ModulesToSaveWrapper và
    # TỰ MỞ KHOÁ (requires_grad=True) ngay cả khi ta không hề yêu cầu, trái
    # với mục 4.3.1 (Head phải đóng băng xuyên suốt giai đoạn Federated LoRA
    # chính). Vì không thể tắt qua cấu hình, ta CHỦ ĐỘNG khoá lại toàn bộ
    # tham số Head ngay sau khi PeftModel được tạo, ghi đè quyết định mặc
    # định của PEFT. `_is_head_param` so khớp theo tên module con ("head",
    # "classifier") nên vẫn đúng với cả hai nhánh mà PEFT tạo ra
    # (*.classifier.original_module.* và *.classifier.modules_to_save.default.*).
    for name, param in model.named_parameters():
        if _is_head_param(name):
            param.requires_grad = False

    # Sanity-check: Head phải thực sự bị đóng băng sau khi đã khoá lại ở trên.
    still_trainable_head = [
        name for name, p in model.named_parameters()
        if _is_head_param(name) and p.requires_grad
    ]
    if still_trainable_head:
        raise RuntimeError(
            "Head vẫn còn requires_grad=True sau khi gắn LoRA, trái với mục "
            f"4.3.1 của đề cương. Tham số còn mở khoá: {still_trainable_head}"
        )

    return model


def count_trainable_parameters(model: nn.Module) -> dict[str, int]:
    """In ra số tham số trainable so với tổng số tham số để sanity-check:
    ở giai đoạn 1, chỉ Head trainable; ở giai đoạn 2, chỉ LoRA (A, B)
    trainable và Head phải bằng 0 tham số mở khoá."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
    }


def get_head_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Trả về dict {tên_param: tensor} chỉ chứa Classification Head (head.*,
    classifier.*), đã .detach().clone().cpu(). Dùng để:
        (a) lưu checkpoint H* sau giai đoạn warm-up (chọn bằng Global
            Validation), và
        (b) nạp H* vào model của giai đoạn Federated LoRA chính qua
            `build_federated_model(cfg, head_state_dict=...)`.
    Hoạt động với cả model chưa bọc PEFT (build_warmup_model) lẫn đã bọc
    PEFT (build_federated_model), vì `_is_head_param` so khớp theo tên
    module con, không phụ thuộc tiền tố PEFT thêm vào.
    """
    state = model.state_dict()
    head_state = {
        name: tensor.detach().clone().cpu()
        for name, tensor in state.items()
        if _is_head_param(name)
    }
    if not head_state:
        raise RuntimeError(
            "Không tìm thấy tham số Head nào trong model để trích xuất. "
            "Kiểm tra lại HEAD_MODULE_NAMES so với model.named_parameters()."
        )
    return head_state


def get_lora_state_dict(model: PeftModel) -> dict[str, torch.Tensor]:
    """
    Trả về dict {tên_param: tensor} chỉ chứa lora_A.* và lora_B.* (đã clone và
    chuyển CPU để an toàn khi serialize hay gửi qua mạng hoặc lưu checkpoint).

    Không bao gồm Head (head.*, classifier.*) vì Head không nằm trong phạm vi
    tổng hợp liên bang ở giai đoạn Federated LoRA chính (mục 4.3.1: Head bị
    đóng băng, không trao đổi giữa Client <-> Server ở giai đoạn này).
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
    chứa 1 phần state (không có backbone, không có Head).
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
    các client (Krum, Trimmed Mean, TC-LoRA) -- u_i = Concat(vec(ΔA_i),
    vec(ΔB_i)) theo mục 3.2/3.4 của đề cương. Các module đó sẽ tự quyết định
    có tách A/B ra so sánh riêng hay gộp chung; hàm này chỉ cung cấp tiện ích
    làm phẳng với thứ tự khoá được sorted nhất quán để mọi client so sánh
    đúng vị trí với nhau.
    """
    sorted_keys = sorted(lora_state.keys())
    return torch.cat([lora_state[k].flatten() for k in sorted_keys])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check build ModernBERT (warm-up + LoRA)")
    parser.add_argument("--num_labels", type=int, default=4, help="4=AG News Redis, 9=DBpedia L1")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = LoRAModelConfig(
        num_labels=args.num_labels,
        lora_rank=args.lora_rank,
        cache_dir=args.cache_dir,
    )
    print(f"alpha/r = {cfg.lora_alpha / cfg.lora_rank} (kỳ vọng = 2 theo mục 4.3.2)")

    print(f"\nĐang tải {cfg.checkpoint} (lần đầu sẽ tải về, các lần sau dùng cache)...")
    tokenizer = build_tokenizer(cfg)

    # --- Giai đoạn 1: warm-up (không LoRA) ---
    warmup_model = build_warmup_model(cfg)
    stats_warmup = count_trainable_parameters(warmup_model)
    print(f"\n[Warm-up] Tham số trainable: {stats_warmup['trainable']:,} / "
          f"{stats_warmup['total']:,} ({stats_warmup['trainable_pct']:.3f}%) "
          f"-- kỳ vọng CHỈ Head mở khoá.")
    head_state = get_head_state_dict(warmup_model)
    print(f"[Warm-up] Số tensor trong head_state_dict: {len(head_state)}")

    # --- Giai đoạn 2: Federated LoRA chính, nạp H* vừa "warm-up" xong ---
    fed_model = build_federated_model(cfg, head_state_dict=head_state)
    stats_fed = count_trainable_parameters(fed_model)
    print(f"\n[Federated LoRA] Tham số trainable: {stats_fed['trainable']:,} / "
          f"{stats_fed['total']:,} ({stats_fed['trainable_pct']:.3f}%) "
          f"-- kỳ vọng CHỈ LoRA A/B mở khoá, Head = 0.")

    print("\n--- Tên module có LoRA (vài dòng đầu) ---")
    for name, _ in fed_model.named_modules():
        if "lora_A" in name or "lora_B" in name:
            print(" ", name)

    lora_state = get_lora_state_dict(fed_model)
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
    fed_model.eval()
    with torch.no_grad():
        out = fed_model(**sample)
    print(f"\nForward pass OK -- logits shape: {out.logits.shape}")

    # Test reproducibility của init_seed (chỉ ảnh hưởng khởi tạo LoRA A)
    cfg2 = LoRAModelConfig(num_labels=args.num_labels, lora_rank=args.lora_rank,
                            init_seed=cfg.init_seed, cache_dir=args.cache_dir)
    fed_model2 = build_federated_model(cfg2, head_state_dict=head_state)
    state1 = get_lora_state_dict(fed_model)
    state2 = get_lora_state_dict(fed_model2)
    same_init = all(torch.equal(state1[k], state2[k]) for k in state1)
    print(f"Cùng init_seed -> cùng khởi tạo LoRA: {same_init}")