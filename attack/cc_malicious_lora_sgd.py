"""
attack/cc_malicious_lora_sgd.py

CC-MaliciousLoRA-SGD: bộ tối ưu hóa cục bộ khảo sát tại Byzantine Client,
theo mục 5.2-5.6 của đề cương. File này gộp 4 thành phần liên quan với nhau:

    5.2  Hàm mất mát đa mục tiêu: L_local = L_clean + λ_poison * L_poison
         -> `combine_local_loss`
    5.3  Bộ tối ưu vận tốc thích nghi MaliciousSGD (điều chỉnh từ Malicious
         Local Optimizer của Fu et al. [9])
         -> `MaliciousSGD` (torch.optim.Optimizer)
    5.4  Clean Reference Update và ràng buộc Cosine (định nghĩa, không phải
         thuật toán tối ưu riêng: shadow training nằm ở federated/, module
         này chỉ cung cấp hàm tính u_mal, u_ref từ state dict đã có)
         -> `compute_delta_state_dict`, `flatten_delta_state_dict`
    5.5  Phép chiếu hướng bảo toàn norm
         -> `project_direction_preserve_norm`
    5.6  Gaussian perturbation và kiểm tra lại ràng buộc
         -> `add_gaussian_and_reproject`

Hàm tổng hợp `finalize_malicious_update()` nối toàn bộ 5.4-5.6 thành một
bước duy nhất: nhận (A_t,B_t), (A_mal,B_mal) sau khi train MaliciousSGD trên
L_local, và (A_ref,B_ref) từ shadow run trên D_clean (tính ở federated/client
Byzantine, không tính trong module này vì cần vòng lặp train/data loader),
trả về adapter cuối cùng sẵn sàng gửi lên Server cùng log chẩn đoán.

Quy ước nguồn ngẫu nhiên (xem data/partition.py, data/poison.py để biết các
nguồn khác): nhiễu Gaussian ξ ở mục 5.6 dùng `torch.Generator` với seed riêng
(`gaussian_seed` trong `CCMaliciousLoRASGDConfig`), khác với seed partition
Client, seed init LoRA A, seed chọn mẫu đầu độc ở data/poison.py. Không
bao giờ gọi `torch.manual_seed()` toàn cục trong module này.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# 5.2. Hàm mất mát đa mục tiêu
# ---------------------------------------------------------------------------

def combine_local_loss(
    loss_clean: torch.Tensor,
    loss_poison: torch.Tensor,
    lambda_poison: float,
) -> torch.Tensor:
    """
    L_local(A,B) = L_clean(A,B) + λ_poison * L_poison(A,B)   (mục 5.2)

    `loss_clean` và `loss_poison` phải là loss trung bình theo số mẫu
    (reduction="mean") tính riêng trên D_clean và D_poison tương ứng (xem
    data/poison.py::build_poisoned_fit_set trả về D_clean/D_poison tách
    biệt chính vì lý do này). Không gộp D_clean và D_poison thành một batch
    rồi tính 1 loss trung bình chung trên D_mal. Cách đó sẽ ngầm định
    trọng số bằng tỷ lệ kích thước (thường ρ_poison nhỏ -> D_poison đóng góp
    rất ít), khác với công thức cộng có trọng số λ_poison tường minh ở đây.
    """
    return loss_clean + lambda_poison * loss_poison


# 5.3. Bộ tối ưu vận tốc thích nghi MaliciousSGD
@dataclass
class MaliciousSGDConfig:
    """Giá trị khởi đầu theo Table 7 của Fu et al. [9] (mục 5.3, bảng mục 8):
    beta=0.9, gamma=1.0, r_min=1.0, r_max=5.0. `lr` (η_m) do nghiên cứu tự
    chọn trong cấu hình tối ưu cục bộ, không lấy từ Fu et al."""
    lr: float
    beta: float = 0.9
    gamma: float = 1.0
    r_min: float = 1.0
    r_max: float = 5.0
    eps_v: float = 1e-8  # ε_v: ngưỡng |v_{s-1}^adj| gần 0 -> đặt r_s=1 (mục 5.3)


class MaliciousSGD(torch.optim.Optimizer):
    """
    Triển khai cấu trúc Algorithm 1 của Fu et al. [9], điều chỉnh để tối ưu
    trực tiếp tham số LoRA θ=(A,B) theo L_local (mục 5.2-5.3).

    Công thức mỗi bước local s (per-coordinate, elementwise):
        v_s^raw = β v_{s-1}^adj + (1-β) g_s
        (bước local đầu tiên của mỗi tham số: v_{-1}^adj := 0, nên
         v_s^raw = (1-β) g_s, và theo đề cương thì "sử dụng v_s^adj=v_s^raw"
         tức chưa áp dụng r_s ở bước này vì chưa có v_{s-1}^adj != 0 để so
         sánh tỷ lệ)
        r_s = clip(1 + γ (v_s^raw / v_{s-1}^adj), r_min, r_max)   [bước >= 2]
        v_s^adj = r_s ⊙ v_{s-1}^adj
        θ_{s+1} = θ_s - η_m v_s^adj

    Ổn định số học: nếu |v_{s-1,j}^adj| <= ε_v thì tọa độ j đặt r_{s,j}=1
    thay vì chia cho giá trị gần 0 (mục 5.3).

    DUY TRÌ TRẠNG THÁI QUA ROUND: "trạng thái vận tốc... được khởi tạo 
    một lần ở đầu quá trình huấn luyện của Byzantine Client,
    sau đó được duy trì qua các local minibatch VÀ các global round; tại đầu
    mỗi round, các tham số A, B được đồng bộ từ Server nhưng trạng thái
    optimizer cục bộ vẫn được giữ." Để đúng yêu cầu này:
        - Dùng lại CÙNG một instance `MaliciousSGD` xuyên suốt toàn bộ quá
          trình huấn luyện của Byzantine Client (không tạo optimizer mới mỗi
          round).
        - Khi đồng bộ (A_t, B_t) mới từ Server, PHẢI nạp giá trị vào ĐÚNG
          cùng object Parameter đang được optimizer theo dõi (vd. qua
          `param.data.copy_(new_value)` hoặc `model.load_state_dict(...,
          strict=False)` như `models/modernbert_lora.py::set_lora_state_dict`
          đã làm -- hàm đó dùng `load_state_dict` nên chỉ COPY GIÁ TRỊ vào
          tensor đang có, KHÔNG thay thế object Parameter, nên state nội bộ
          của optimizer (khoá theo `id(param)`) vẫn hợp lệ qua các round).
          KHÔNG được build lại model/optimizer mới mỗi round vì sẽ làm mất
          toàn bộ lịch sử vận tốc, trái với chính sách triển khai của đề
          cương.
    """

    def __init__(
        self,
        params,
        lr: float,
        beta: float = 0.9,
        gamma: float = 1.0,
        r_min: float = 1.0,
        r_max: float = 5.0,
        eps_v: float = 1e-8,
    ):
        if lr <= 0:
            raise ValueError(f"lr (η_m) phải > 0, nhận {lr}")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta phải trong [0, 1), nhận {beta}")
        if r_min > r_max:
            raise ValueError(f"r_min ({r_min}) phải <= r_max ({r_max})")
        defaults = dict(lr=lr, beta=beta, gamma=gamma, r_min=r_min, r_max=r_max, eps_v=eps_v)
        super().__init__(params, defaults)

    @classmethod
    def from_config(cls, params, cfg: MaliciousSGDConfig) -> "MaliciousSGD":
        return cls(params, lr=cfg.lr, beta=cfg.beta, gamma=cfg.gamma,
                   r_min=cfg.r_min, r_max=cfg.r_max, eps_v=cfg.eps_v)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            gamma = group["gamma"]
            r_min = group["r_min"]
            r_max = group["r_max"]
            eps_v = group["eps_v"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if "v_adj" not in state:
                    # Bước local ĐẦU TIÊN của tham số này: v_{-1}^adj := 0
                    # (khởi tạo EMA từ không) -> v_s^raw = (1-beta)*g; theo
                    # đề cương "Ở bước local đầu tiên, sử dụng v_s^adj=v_s^raw".
                    v_raw = (1.0 - beta) * g
                    v_adj = v_raw.clone()
                else:
                    v_prev_adj = state["v_adj"]
                    v_raw = beta * v_prev_adj + (1.0 - beta) * g

                    safe_mask = v_prev_adj.abs() > eps_v
                    ratio = torch.zeros_like(v_prev_adj)
                    ratio[safe_mask] = v_raw[safe_mask] / v_prev_adj[safe_mask]

                    r = 1.0 + gamma * ratio
                    r = torch.clamp(r, min=r_min, max=r_max)
                    # Toạ độ |v_{s-1}^adj| <= eps_v: đặt r=1 thay vì dùng giá
                    # trị clip tính từ phép chia không an toàn (mục 5.3).
                    r = torch.where(safe_mask, r, torch.ones_like(r))

                    v_adj = r * v_prev_adj

                state["v_adj"] = v_adj
                p.data.add_(v_adj, alpha=-lr)

        return loss

    def reset_velocity_state(self) -> None:
        """Xoá toàn bộ trạng thái vận tốc đã lưu (v_adj). CHỈ dùng cho mục
        đích ablation (vd. so sánh "có reset mỗi round" vs "không reset" như
        một biến thể phụ) -- KHÔNG gọi hàm này trong luồng thực nghiệm chính,
        vì đề cương yêu cầu vận tốc được duy trì xuyên suốt (xem docstring
        class ở trên)."""
        self.state = {}



# 5.4 (một phần). Tiện ích xây u_mal, u_ref từ state dict LoRA
def compute_delta_state_dict(
    state_before: dict[str, torch.Tensor],
    state_after: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Δθ = θ_after - θ_before theo từng tensor khớp tên. Dùng để tính
    ΔA = A_local - A_t, ΔB = B_local - B_t (mục 3.1, 3.4, 5.4) cho cả nhánh
    u_mal (state_after=trạng thái sau MaliciousSGD) và nhánh u_ref
    (state_after=trạng thái sau shadow run trên D_clean)."""
    if set(state_before.keys()) != set(state_after.keys()):
        missing = set(state_before.keys()) ^ set(state_after.keys())
        raise KeyError(f"state_before và state_after lệch khoá: {missing}")
    return {k: (state_after[k] - state_before[k]) for k in state_before}


def flatten_delta_state_dict(delta_state: dict[str, torch.Tensor]) -> torch.Tensor:
    """u = Concat(vec(ΔA), vec(ΔB)) theo thứ tự khoá đã sorted (mục 3.2, 3.4,
    5.4). Thứ tự sorted-key này PHẢI khớp với
    `models/modernbert_lora.py::lora_state_dict_to_flat_vector` để vector
    u_mal/u_ref ở đây tương thích vị trí tọa độ với vector mà các bộ tổng hợp
    (Krum, Trimmed Mean, TC-LoRA) sử dụng."""
    sorted_keys = sorted(delta_state.keys())
    return torch.cat([delta_state[k].reshape(-1) for k in sorted_keys])


def unflatten_to_state_dict(
    flat: torch.Tensor,
    reference_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Chiều ngược lại của `flatten_delta_state_dict`: tách vector 1-D trở
    lại thành dict {key: tensor} theo đúng shape của `reference_state` (dùng
    `reference_state` chỉ để lấy tên khoá + shape theo thứ tự sorted, không
    đọc giá trị của nó)."""
    sorted_keys = sorted(reference_state.keys())
    out: dict[str, torch.Tensor] = {}
    cursor = 0
    for k in sorted_keys:
        shape = reference_state[k].shape
        numel = reference_state[k].numel()
        out[k] = flat[cursor:cursor + numel].reshape(shape).clone()
        cursor += numel
    if cursor != flat.numel():
        raise ValueError(
            f"Kích thước flat vector ({flat.numel()}) không khớp tổng kích "
            f"thước reference_state ({cursor})."
        )
    return out


# 5.5. Phép chiếu hướng bảo toàn norm
@dataclass
class CosineProjectionConfig:
    """τ là siêu tham số khảo sát (0.80, 0.85, 0.90 -- mục 5.4, bảng mục 8).
    Các ngưỡng eps_* là hằng số số học nhỏ để xử lý suy biến (mục 5.5-5.6),
    không phải siêu tham số khảo sát."""
    tau: float
    eps_ref: float = 1e-8   # ||u_ref|| gần 0 -> không có hướng tham chiếu hợp lệ
    eps_rho: float = 1e-8   # ||u_mal|| hoặc ||u_candidate|| gần 0
    eps_q: float = 1e-8     # ||q|| gần 0 -> trường hợp suy biến (gần đối song)

    def __post_init__(self):
        if not (0.0 < self.tau < 1.0):
            raise ValueError(f"tau phải trong (0, 1), nhận {self.tau}")


def _pick_non_parallel_basis(r: torch.Tensor) -> torch.Tensor:
    """Chọn 1 vector cơ sở chuẩn e_k KHÔNG song song với r: lấy k là toạ độ
    có |r_k| NHỎ NHẤT (toạ độ đóng góp ít nhất vào hướng r, nên e_k gần trực
    giao với r nhất, giảm nguy cơ suy biến ở bước trực giao hoá tiếp theo).
    Đây là quy tắc chọn e CỤ THỂ do đề tài quyết định triển khai -- đề cương
    (mục 5.5) chỉ yêu cầu "e không song song với r", không chỉ định thuật
    toán chọn; quy tắc argmin này là tất định (không cần RNG), đảm bảo tái
    lập hoàn toàn."""
    k = int(torch.argmin(r.abs()).item())
    e = torch.zeros_like(r)
    e[k] = 1.0
    return e


def _project_unit_onto_cap(
    unit_vec: torch.Tensor,
    r: torch.Tensor,
    tau: float,
    eps_q: float,
) -> torch.Tensor:
    """
    Trả về một vector ĐƠN VỊ nằm trên biên nón Cosine quanh hướng `r` với
    ngưỡng `tau`, dùng chung cho cả bước 5.5 (chiếu u_mal) và bước 5.6
    (chiếu lại u_candidate sau nhiễu Gaussian). Về đại số, công thức tính
    q_hat ở hai bước giống hệt nhau một khi đầu vào đã được chuẩn hoá thành
    vector đơn vị (chuẩn hoá không đổi HƯỚNG nên q_hat không đổi theo việc
    đầu vào có được nhân với ρ hay không) -- dùng chung một hàm để tránh
    lặp code và đảm bảo nhất quán giữa hai bước.
    """
    c = torch.dot(unit_vec, r)
    if c.item() >= tau:
        return unit_vec

    q = unit_vec - c * r
    q_norm = torch.linalg.norm(q)
    if q_norm.item() > eps_q:
        q_hat = q / q_norm
    else:
        e = _pick_non_parallel_basis(r)
        q_perp = e - torch.dot(e, r) * r
        q_perp_norm = torch.linalg.norm(q_perp)
        if q_perp_norm.item() <= eps_q:
            raise RuntimeError(
                "Không tìm được vector trực giao hợp lệ với r (q_perp suy biến "
                "cả sau khi đổi cơ sở) -- kiểm tra lại r có phải vector không hay không."
            )
        q_hat = q_perp / q_perp_norm

    tau_t = torch.as_tensor(tau, dtype=unit_vec.dtype, device=unit_vec.device)
    coeff = torch.sqrt(torch.clamp(1.0 - tau_t * tau_t, min=0.0))
    return tau_t * r + coeff * q_hat


@torch.no_grad()
def project_direction_preserve_norm(
    u_mal: torch.Tensor,
    u_ref: torch.Tensor,
    cfg: CosineProjectionConfig,
) -> tuple[torch.Tensor, dict]:
    """
    Mục 5.5. Chiếu u_mal lên biên nón Cosine {u: ||u||=ρ, Sim_cos(u,u_ref)>=τ}
    nếu chưa thoả, bảo toàn norm ρ=||u_mal||.

    Trả về (u_projected, info) với `info` ghi lại các trường hợp suy biến và
    cos similarity trước/sau để log kiểm tra (mục 5.6: "Các trường hợp suy
    biến, norm, Cosine sau chiếu... đều được ghi log").
    """
    info: dict = {
        "skipped_no_ref": False,
        "kept_zero": False,
        "already_satisfied": False,
        "cos_sim_before": None,
        "cos_sim_after": None,
    }

    ref_norm = torch.linalg.norm(u_ref)
    if ref_norm.item() <= cfg.eps_ref:
        # "Nếu ||u_ref||₂ gần bằng 0, không có hướng tham chiếu hợp lệ và
        # bước chiếu được bỏ qua" (mục 5.5).
        info["skipped_no_ref"] = True
        return u_mal.clone(), info

    r = u_ref / ref_norm
    rho = torch.linalg.norm(u_mal)
    if rho.item() <= cfg.eps_rho:
        # "Nếu ρ gần bằng 0, cập nhật được giữ bằng vector không" (mục 5.5).
        info["kept_zero"] = True
        return torch.zeros_like(u_mal), info

    unit_mal = u_mal / rho
    c = torch.dot(unit_mal, r)
    info["cos_sim_before"] = c.item()

    if c.item() >= cfg.tau:
        info["already_satisfied"] = True
        info["cos_sim_after"] = c.item()
        return u_mal.clone(), info

    unit_proj = _project_unit_onto_cap(unit_mal, r, cfg.tau, cfg.eps_q)
    u_projected = rho * unit_proj
    info["cos_sim_after"] = torch.dot(unit_proj, r).item()
    return u_projected, info


# 5.6. Gaussian perturbation và kiểm tra lại ràng buộc
@torch.no_grad()
def add_gaussian_and_reproject(
    u_projected: torch.Tensor,
    u_ref: torch.Tensor,
    cfg: CosineProjectionConfig,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Mục 5.6. ξ ~ N(0, σ² I_d); u_candidate = u_projected + ξ; sau đó chiếu
    lại lên C_{τ,ρ} = {u: ||u||=ρ, Sim_cos(u,u_ref)>=τ} với ρ=||u_projected||
    GIỮ NGUYÊN từ bước 5.5 (không tính lại ρ từ u_candidate).

    `generator`: torch.Generator đã set seed RIÊNG cho nhiễu Gaussian (xem
    docstring đầu file) -- PHẢI truyền vào, không dùng RNG toàn cục, để tái
    lập độc lập với các nguồn ngẫu nhiên khác trong hệ thống.
    """
    info: dict = {
        "skipped_no_ref": False,
        "degenerate_candidate": False,
        "cos_sim_candidate": None,
        "cos_sim_final": None,
    }

    rho = torch.linalg.norm(u_projected)
    noise = torch.randn(
        u_projected.shape, generator=generator,
        dtype=u_projected.dtype, device=u_projected.device,
    ) * sigma
    u_candidate = u_projected + noise

    ref_norm = torch.linalg.norm(u_ref)
    if ref_norm.item() <= cfg.eps_ref:
        # Nhất quán với 5.5: nếu không có hướng tham chiếu hợp lệ thì không
        # chiếu lại, trả thẳng u_candidate (đã cộng nhiễu, chưa ràng buộc).
        info["skipped_no_ref"] = True
        return u_candidate, info
    r = u_ref / ref_norm

    cand_norm = torch.linalg.norm(u_candidate)
    if cand_norm.item() <= cfg.eps_rho:
        # "nếu u_candidate gần bằng không thì chọn hướng r" (mục 5.6) --
        # khác với 5.5 (giữ vector KHÔNG): ở đây ρ đã được xác lập hợp lệ từ
        # bước 5.5 (khác 0 vì đã qua nhánh kept_zero ở trên), nên luôn trả
        # ρ*r thay vì vector không.
        info["degenerate_candidate"] = True
        info["cos_sim_final"] = 1.0
        return rho * r, info

    v = u_candidate / cand_norm
    c = torch.dot(v, r)
    info["cos_sim_candidate"] = c.item()

    if c.item() >= cfg.tau:
        info["cos_sim_final"] = c.item()
        return rho * v, info

    unit_final = _project_unit_onto_cap(v, r, cfg.tau, cfg.eps_q)
    info["cos_sim_final"] = torch.dot(unit_final, r).item()
    return rho * unit_final, info


# Hàm tổng hợp: nối 5.4 (delta) + 5.5 (chiếu) + 5.6 (nhiễu + chiếu lại)
@dataclass
class CCMaliciousLoRASGDConfig:
    cosine: CosineProjectionConfig
    lambda_poison: float      # mục 5.2
    gaussian_sigma: float     # σ, mục 5.6 -- siêu tham số khảo sát
    gaussian_seed: int        # seed RIÊNG cho nhiễu Gaussian ξ (xem docstring đầu file)


@torch.no_grad()
def finalize_malicious_update(
    state_t: dict[str, torch.Tensor],
    state_mal: dict[str, torch.Tensor],
    state_ref: dict[str, torch.Tensor],
    cfg: CCMaliciousLoRASGDConfig,
) -> tuple[dict[str, torch.Tensor], dict]:
    """
    Nối mục 5.4 (tính u_mal, u_ref) + 5.5 (chiếu bảo toàn norm) + 5.6 (nhiễu
    Gaussian rồi chiếu lại) thành một bước duy nhất.

    Tham số:
        state_t:    (A_t, B_t) -- trạng thái global Server phát tán đầu round.
        state_mal:  (A_mal, B_mal) -- sau khi Byzantine Client train
                    MaliciousSGD trên L_local = L_clean + λ_poison*L_poison
                    (train loop nằm ở federated/, module này không chạy
                    training).
        state_ref:  (A_ref, B_ref) -- Clean Reference Update: kết quả shadow
                    training CHỈ trên D_clean, cùng khởi tạo (A_t,B_t), cùng
                    số local step/batch schedule/optimizer config với nhánh
                    chính (mục 5.4). Shadow training cũng nằm ở federated/.
        cfg:        CCMaliciousLoRASGDConfig.

    Trả về (final_state, diagnostics):
        final_state: {key: A_t[key]+ΔA_final, ...} -- adapter hoàn chỉnh sẵn
            sàng gửi lên Server (A_t+ΔA_mal, B_t+ΔB_mal theo ký hiệu mục 5.6).
        diagnostics: dict ghi log norm, cos similarity trước/sau chiếu, sau
            nhiễu, và các cờ trường hợp suy biến -- đúng yêu cầu "Các trường
            hợp suy biến, norm, Cosine sau chiếu và sau nhiễu đều được ghi
            log để kiểm tra tính đúng đắn của triển khai" (cuối mục 5.6).
    """
    delta_mal = compute_delta_state_dict(state_t, state_mal)
    delta_ref = compute_delta_state_dict(state_t, state_ref)

    u_mal = flatten_delta_state_dict(delta_mal)
    u_ref = flatten_delta_state_dict(delta_ref)

    u_projected, info_proj = project_direction_preserve_norm(u_mal, u_ref, cfg.cosine)

    generator = torch.Generator(device=u_projected.device)
    generator.manual_seed(cfg.gaussian_seed)
    u_final, info_noise = add_gaussian_and_reproject(
        u_projected, u_ref, cfg.cosine, cfg.gaussian_sigma, generator=generator,
    )

    delta_final = unflatten_to_state_dict(u_final, reference_state=delta_mal)
    final_state = {k: state_t[k] + delta_final[k] for k in state_t}

    diagnostics = {
        "u_mal_norm": torch.linalg.norm(u_mal).item(),
        "u_ref_norm": torch.linalg.norm(u_ref).item(),
        "u_final_norm": torch.linalg.norm(u_final).item(),
        **{f"proj_{k}": v for k, v in info_proj.items()},
        **{f"noise_{k}": v for k, v in info_noise.items()},
    }
    return final_state, diagnostics


if __name__ == "__main__":
    torch.manual_seed(0)  # chỉ dùng cho self-test, không phải seed của pipeline thật

    # --- Test MaliciousSGD: velocity formula đúng tay ---
    print("=== Test MaliciousSGD ===")
    p = torch.nn.Parameter(torch.zeros(5))
    opt = MaliciousSGD([p], lr=0.1, beta=0.9, gamma=1.0, r_min=1.0, r_max=5.0)

    g1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    p.grad = g1.clone()
    opt.step()
    expected_v1 = (1 - 0.9) * g1  # bước đầu: v_adj = v_raw = (1-beta)*g
    expected_p1 = -0.1 * expected_v1
    assert torch.allclose(opt.state[p]["v_adj"], expected_v1), "Sai công thức bước local đầu tiên"
    assert torch.allclose(p.data, expected_p1), "Sai cập nhật tham số bước đầu"
    print(f"  Bước 1 OK: v_adj={opt.state[p]['v_adj'].tolist()}")

    g2 = torch.tensor([1.0, -2.0, 0.0, 4.0, -5.0])
    v_prev = expected_v1.clone()
    p.grad = g2.clone()
    opt.step()
    v_raw2 = 0.9 * v_prev + 0.1 * g2
    ratio2 = v_raw2 / v_prev
    r2 = torch.clamp(1.0 + 1.0 * ratio2, 1.0, 5.0)
    v_adj2_expected = r2 * v_prev
    assert torch.allclose(opt.state[p]["v_adj"], v_adj2_expected, atol=1e-6), "Sai công thức bước local thứ hai"
    print(f"  Bước 2 OK: v_adj={opt.state[p]['v_adj'].tolist()}")
    print(f"  r_s (bước 2) = {r2.tolist()} (phải trong [1.0, 5.0])")

    # --- Test project_direction_preserve_norm + add_gaussian_and_reproject ---
    print("\n=== Test chiếu Cosine (mục 5.5-5.6), bằng torch ===")
    cfg_cos = CosineProjectionConfig(tau=0.85)
    d = 300
    u_ref = torch.randn(d)
    u_mal = torch.randn(d)
    u_mal = u_mal - (torch.dot(u_mal, u_ref) / torch.dot(u_ref, u_ref)) * u_ref  # ép gần trực giao

    rho_before = torch.linalg.norm(u_mal).item()
    u_proj, info_p = project_direction_preserve_norm(u_mal, u_ref, cfg_cos)
    rho_after = torch.linalg.norm(u_proj).item()
    cos_after = torch.dot(u_proj, u_ref / torch.linalg.norm(u_ref)).item()
    print(f"  ||u_mal||={rho_before:.4f}  ||u_proj||={rho_after:.4f} (phải bằng nhau)")
    print(f"  cos_sim(u_proj,u_ref)={cos_after:.6f} (phải == tau={cfg_cos.tau})")
    assert abs(rho_before - rho_after) < 1e-4
    assert abs(cos_after - cfg_cos.tau) < 1e-4

    gen = torch.Generator()
    gen.manual_seed(2026)
    for sigma in (0.01, 0.1, 1.0):
        u_final, info_n = add_gaussian_and_reproject(u_proj, u_ref, cfg_cos, sigma, generator=gen)
        norm_final = torch.linalg.norm(u_final).item()
        cos_final = torch.dot(u_final, u_ref / torch.linalg.norm(u_ref)).item()
        print(f"  sigma={sigma:5.2f}  ||u_final||={norm_final:.4f} (kỳ vọng={rho_after:.4f})  "
              f"cos_sim_final={cos_final:.6f} (kỳ vọng >= {cfg_cos.tau})")
        assert abs(norm_final - rho_after) < 1e-3
        assert cos_final >= cfg_cos.tau - 1e-4

    # --- Test finalize_malicious_update end-to-end với state dict LoRA giả lập ---
    print("\n=== Test finalize_malicious_update (end-to-end) ===")
    A_shape, B_shape = (8, 32), (32, 8)  # r=8 ví dụ
    state_t = {"lora_A": torch.randn(A_shape) * 0.01, "lora_B": torch.zeros(B_shape)}
    state_mal = {"lora_A": state_t["lora_A"] + torch.randn(A_shape) * 0.05,
                 "lora_B": state_t["lora_B"] + torch.randn(B_shape) * 0.05}
    state_ref = {"lora_A": state_t["lora_A"] + torch.randn(A_shape) * 0.03,
                 "lora_B": state_t["lora_B"] + torch.randn(B_shape) * 0.03}

    cfg_full = CCMaliciousLoRASGDConfig(
        cosine=CosineProjectionConfig(tau=0.85),
        lambda_poison=1.0,
        gaussian_sigma=0.05,
        gaussian_seed=999,
    )
    final_state, diag = finalize_malicious_update(state_t, state_mal, state_ref, cfg_full)
    print(f"  Shapes: lora_A={final_state['lora_A'].shape}, lora_B={final_state['lora_B'].shape}")
    print(f"  Diagnostics: {diag}")
    assert final_state["lora_A"].shape == A_shape
    assert final_state["lora_B"].shape == B_shape
    assert diag["noise_cos_sim_final"] is None or diag["noise_cos_sim_final"] >= cfg_full.cosine.tau - 1e-3

    # Reproducibility theo gaussian_seed
    final_state2, diag2 = finalize_malicious_update(state_t, state_mal, state_ref, cfg_full)
    same = all(torch.allclose(final_state[k], final_state2[k]) for k in final_state)
    print(f"  Reproducible với cùng gaussian_seed: {same}")
    assert same

    print("\nTẤT CẢ TEST PASS.")
