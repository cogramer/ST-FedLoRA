"""
federated/server.py

Server orchestration: Phase 1 Federated Classification Head Warm-up (mục
4.3.1) và Phase 2 Federated LoRA chính, mỗi cơ chế tổng hợp là 1 "nhánh thí
nghiệm độc lập" (mục 7.1). Server KHÔNG tự cài lại toán học tổng hợp --
gọi federated.aggregation.run_aggregator cho Phase 2.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW

from attack.cc_malicious_lora_sgd import compute_delta_state_dict
from federated.aggregation import run_aggregator
from federated.aggregation.base import weighted_average_state_dicts
from federated.aggregation.tc_lora import consistency_matrix
from federated.client import (
    BenignClientConfig,
    ByzantineClientConfig,
    compute_validation_descent_direction,
    train_benign_client,
    train_byzantine_client,
)
from federated.dataset import make_dataloader
from models.modernbert_lora import get_head_state_dict

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm luôn nên có trong môi trường Colab
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        """Fallback không phụ thuộc tqdm: trả về chính iterable, bỏ qua args."""
        return iterable if iterable is not None else range(0)


# ============================================================
# Phase 1: Federated Classification Head Warm-up (mục 4.3.1)
# ============================================================


@dataclass
class WarmupConfig:
    """
    Đề cương chỉ yêu cầu "quy tắc dừng sớm đã định trước" mà không chỉ định
    thuật toán -- patience-based early stopping trên Global Validation là
    THIẾT KẾ của người triển khai, cần ghi rõ trong báo cáo thí nghiệm.
    """

    max_rounds: int = 20
    patience: int = 3
    local_epochs: int = 3
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_length: int = 256


def _train_head_one_client(
    model, tokenizer, head_state, df_fit, cfg: WarmupConfig, shuffle_seed: int, device: str,
    client_tag: str = "",
):
    """
    1 Client trong warm-up: Encoder đóng băng, LoRA CHƯA gắn, chỉ Head
    được tối ưu bằng AdamW trên D_i^fit (mục 4.3.1). Model truyền vào phải
    ở chế độ models.modernbert_lora.build_warmup_model().

    client_tag: chuỗi hiển thị trong log/tqdm (vd "round 0 | client 2") chỉ
        để debug tiến trình, KHÔNG ảnh hưởng tới kết quả huấn luyện.
    """
    model.load_state_dict(head_state, strict=False)
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
            desc=f"[warmup-head {client_tag}] epoch {epoch + 1}/{cfg.local_epochs}",
            leave=False,
        )
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            out = model(**batch, labels=labels)
            out.loss.backward()
            optimizer.step()

            running_loss += out.loss.item()
            num_batches += 1
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

        avg_loss = running_loss / num_batches if num_batches > 0 else float("nan")
        print(f"[warmup-head {client_tag}] epoch {epoch + 1}/{cfg.local_epochs} xong, loss trung bình = {avg_loss:.4f}")

    return get_head_state_dict(model)


def run_head_warmup(
    model,
    tokenizer,
    client_fit_dfs: List[Any],
    client_num_samples: List[int],
    df_global_val,
    cfg: WarmupConfig,
    eval_fn: Callable[[Any, Any, Any, str], float],
    round_seed_offset: int,
    device: str = "cuda",
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    Phase 1 (mục 4.3.1): FedAvg trên Head, trọng số theo số mẫu
    p_i=n_i^fit/n_fit (công thức mục 3.1 áp dụng cho tham số Head thay vì
    LoRA). Dừng sớm bằng Global Validation: `patience` round liên tiếp
    không cải thiện accuracy -> dừng, GIỮ checkpoint Head TỐT NHẤT đã thấy
    (không phải checkpoint ở round dừng).

    Không có Client Byzantine hoặc mẫu Trigger ở giai đoạn này (mục 4.3.1):
    `client_fit_dfs` phải là 5 DataFrame "sạch" (với Client sẽ-là-Byzantine
    ở Phase 2, dùng D_clean của nó, KHÔNG dùng D_poison).

    eval_fn(model, df_global_val, tokenizer, device) -> float (accuracy);
    truyền 1 hàm tạm thời trong lúc eval/ (mục 7.2) chưa viết.

    Trả về (head_state_tốt_nhất, diagnostics lịch sử theo round).
    """
    head_state = get_head_state_dict(model)
    best_head_state = copy.deepcopy(head_state)
    best_val_acc = -1.0
    rounds_without_improvement = 0
    history: List[Dict] = []

    total_samples = sum(client_num_samples)
    weights = [n / total_samples for n in client_num_samples]
    num_clients = len(client_fit_dfs)

    print(f"[warmup] bắt đầu: tối đa {cfg.max_rounds} round, {num_clients} client, patience={cfg.patience}")

    for t in range(cfg.max_rounds):
        print(f"[warmup] === round {t + 1}/{cfg.max_rounds} ===")
        client_head_states = []
        for i, df_fit in enumerate(client_fit_dfs):
            seed = round_seed_offset + t * 1000 + i
            print(f"[warmup]   client {i + 1}/{num_clients} (n={client_num_samples[i]}) bắt đầu huấn luyện...")
            client_head_states.append(
                _train_head_one_client(
                    model, tokenizer, head_state, df_fit, cfg, seed, device,
                    client_tag=f"round {t + 1} | client {i}",
                )
            )

        head_state = weighted_average_state_dicts(client_head_states, weights)

        model.load_state_dict(head_state, strict=False)
        val_acc = eval_fn(model, df_global_val, tokenizer, device)
        history.append({"round": t, "val_acc": val_acc})
        print(f"[warmup] round {t + 1}/{cfg.max_rounds} xong -- val_acc = {val_acc:.4f} (best hiện tại = {best_val_acc:.4f})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_state = copy.deepcopy(head_state)
            rounds_without_improvement = 0
            print(f"[warmup]   -> cải thiện, lưu checkpoint Head tốt nhất (val_acc={best_val_acc:.4f})")
        else:
            rounds_without_improvement += 1
            print(f"[warmup]   -> không cải thiện ({rounds_without_improvement}/{cfg.patience} round liên tiếp)")
            if rounds_without_improvement >= cfg.patience:
                print(f"[warmup] dừng sớm tại round {t + 1} (patience={cfg.patience} đã đạt)")
                break

    print(f"[warmup] hoàn tất: best_val_acc = {best_val_acc:.4f}, tổng {len(history)} round đã chạy")

    diagnostics = {"history": history, "best_val_acc": best_val_acc, "stopped_at_round": len(history) - 1}
    return best_head_state, diagnostics


# ============================================================
# Phase 2: Federated LoRA chính, 1 nhánh = 1 cơ chế tổng hợp (mục 7.1, 6.2)
# ============================================================


@dataclass
class FederatedLoRAConfig:
    mechanism: str  # "fedavg" | "krum" | "multi_krum" | "trimmed_mean" | "tc_lora"
    num_rounds: int = 30
    byzantine_index: int = 4  # vị trí Client Byzantine trong client_deltas (0-based)
    f: int = 1
    beta_trim: float = 0.2
    multi_krum_m: Optional[int] = None  # None -> mặc định N-f (xem federated/aggregation/krum.py)
    tc_lora_eps: float = 1e-8


class FederatedLoRAExperiment:
    """
    1 nhánh thí nghiệm độc lập (mục 7.1 bước 3-4) cho 1 cơ chế tổng hợp.

    QUYẾT ĐỊNH THIẾT KẾ: đề cương nói "mỗi phương pháp được chạy như một
    nhánh thí nghiệm độc lập từ cùng một trạng thái global đầu vòng và
    cùng tập Client submissions", nhưng không nói rõ liệu 5 cơ chế có chia
    sẻ DUY NHẤT một quỹ đạo (A_t,B_t) toàn cục xuyên suốt 30 round hay mỗi
    cơ chế tự diễn tiến theo quỹ đạo riêng. Ở đây chọn: 5 instance của lớp
    này (1 mỗi cơ chế) xuất phát từ CÙNG (A_0,B_0), CÙNG partition dữ liệu,
    CÙNG cấu hình tấn công/seed (nên vòng 0 cho ra client submissions giống
    hệt nhau giữa các nhánh), nhưng mỗi instance TỰ GIỮ (A_t,B_t) riêng và
    để nó diễn tiến theo đúng cơ chế tổng hợp của chính nó từ round 1 trở
    đi -- đây là cách thiết lập chuẩn trong các nghiên cứu Byzantine-robust
    FL gốc (Blanchard et al. [18], Yin et al. [19]) để so sánh công bằng
    nhiều bộ tổng hợp trên cùng điều kiện khởi tạo.
    """

    def __init__(
        self,
        model,
        tokenizer,
        initial_lora_state: Dict[str, torch.Tensor],
        benign_client_dfs: List[Any],
        byzantine_df_clean,
        byzantine_df_poison,
        trust_dfs: List[Any],
        num_samples: List[int],
        benign_cfg: BenignClientConfig,
        byzantine_cfg: ByzantineClientConfig,
        malicious_optimizer,
        agg_cfg: FederatedLoRAConfig,
        device: str = "cuda",
    ):
        """
        benign_client_dfs: list[df_fit] của 4 Client lành tính (KHÔNG gồm
            Byzantine), theo đúng thứ tự Client 0..3 trước khi Byzantine
            được chèn vào vị trí agg_cfg.byzantine_index.
        trust_dfs, num_samples: list[*] độ dài N=5, THEO ĐÚNG THỨ TỰ CUỐI
            (sau khi Byzantine ở đúng vị trí agg_cfg.byzantine_index) --
            tức cùng thứ tự với client_deltas được dựng trong _run_one_round.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.lora_state = copy.deepcopy(initial_lora_state)
        self.benign_client_dfs = benign_client_dfs
        self.byzantine_df_clean = byzantine_df_clean
        self.byzantine_df_poison = byzantine_df_poison
        self.trust_dfs = trust_dfs
        self.num_samples = num_samples
        self.benign_cfg = benign_cfg
        self.byzantine_cfg = byzantine_cfg
        self.malicious_optimizer = malicious_optimizer
        self.agg_cfg = agg_cfg
        self.device = device
        self.round_logs: List[Dict] = []

    def _run_one_round(self, t: int) -> Dict:
        state_t = copy.deepcopy(self.lora_state)
        client_local_states: List[Dict[str, torch.Tensor]] = []

        print(f"[phase2:{self.agg_cfg.mechanism}] === round {t + 1}/{self.agg_cfg.num_rounds} ===")

        # 4 Client lành tính, AdamW (mục 4.4).
        for i, df_fit in enumerate(self.benign_client_dfs):
            shuffle_seed = 10_000 + t * 100 + i
            print(f"[phase2:{self.agg_cfg.mechanism}]   benign client {i + 1}/{len(self.benign_client_dfs)} bắt đầu...")
            local_state = train_benign_client(
                self.model, self.tokenizer, state_t, df_fit, self.benign_cfg,
                shuffle_seed=shuffle_seed, device=self.device,
                client_tag=f"{self.agg_cfg.mechanism} | round {t + 1} | client {i}",
            )
            client_local_states.append(local_state)

        # 1 Client Byzantine, CC-MaliciousLoRA-SGD (mục 5.1-5.6), chèn đúng
        # vị trí byzantine_index để khớp thứ tự num_samples/trust_dfs.
        print(f"[phase2:{self.agg_cfg.mechanism}]   byzantine client bắt đầu...")
        byzantine_local_state, byzantine_diag = train_byzantine_client(
            self.model, self.tokenizer, state_t,
            self.byzantine_df_clean, self.byzantine_df_poison,
            self.byzantine_cfg, self.malicious_optimizer,
            shuffle_seed_clean=20_000 + t, shuffle_seed_poison=30_000 + t,
            shuffle_seed_shadow=40_000 + t, device=self.device,
            round_tag=f"{self.agg_cfg.mechanism} | round {t + 1}",
        )
        client_local_states.insert(self.agg_cfg.byzantine_index, byzantine_local_state)

        # Server: delta mỗi Client so với state_t (mục 3.4): Δθ_i = θ_i^local − θ_t.
        client_deltas = [compute_delta_state_dict(state_t, s) for s in client_local_states]

        aggregated_delta, agg_diag = self._run_aggregation(state_t, client_deltas)

        if aggregated_delta is None:
            # TC-LoRA: round bị bỏ (mục 6.2 bước 4 cuối) -> giữ nguyên (A_t,B_t).
            round_accepted = False
        else:
            round_accepted = True
            for key in self.lora_state:
                self.lora_state[key] = state_t[key] + aggregated_delta[key]

        round_log = {
            "round": t,
            "mechanism": self.agg_cfg.mechanism,
            "round_accepted": round_accepted,
            "aggregation_diagnostics": agg_diag,
            "byzantine_diagnostics": byzantine_diag,
        }
        self.round_logs.append(round_log)
        print(
            f"[phase2:{self.agg_cfg.mechanism}] round {t + 1}/{self.agg_cfg.num_rounds} xong "
            f"(round_accepted={round_accepted})"
        )
        return round_log

    def _run_aggregation(
        self, state_t: Dict[str, torch.Tensor], client_deltas: List[Dict[str, torch.Tensor]]
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict]:
        mechanism = self.agg_cfg.mechanism
        kwargs: Dict[str, Any] = {}

        if mechanism == "fedavg":
            kwargs["num_samples"] = self.num_samples
        elif mechanism == "krum":
            kwargs["f"] = self.agg_cfg.f
        elif mechanism == "multi_krum":
            kwargs["f"] = self.agg_cfg.f
            kwargs["m"] = self.agg_cfg.multi_krum_m
        elif mechanism == "trimmed_mean":
            kwargs["beta"] = self.agg_cfg.beta_trim
        elif mechanism == "tc_lora":
            # Bước 2-3 (mục 6.2): mỗi Client tính d_i^val TẠI state_t bằng
            # D_i^trust RIÊNG của mình, Server dựng ma trận consistency.
            validation_directions = [
                compute_validation_descent_direction(
                    self.model, self.tokenizer, state_t, df_trust,
                    batch_size=self.benign_cfg.batch_size, device=self.device,
                )
                for df_trust in self.trust_dfs
            ]
            score_matrix = consistency_matrix(client_deltas, validation_directions, eps=self.agg_cfg.tc_lora_eps)
            kwargs["score_matrix"] = score_matrix
            kwargs["eps"] = self.agg_cfg.tc_lora_eps
        else:
            raise ValueError(f"FederatedLoRAExperiment: mechanism '{mechanism}' không được hỗ trợ.")

        return run_aggregator(mechanism, client_deltas, **kwargs)

    def run(self) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        print(f"[phase2:{self.agg_cfg.mechanism}] bắt đầu nhánh: {self.agg_cfg.num_rounds} round")
        for t in range(self.agg_cfg.num_rounds):
            self._run_one_round(t)
        print(f"[phase2:{self.agg_cfg.mechanism}] hoàn tất nhánh ({len(self.round_logs)} round đã chạy)")
        return self.lora_state, self.round_logs
