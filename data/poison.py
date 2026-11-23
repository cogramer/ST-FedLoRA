"""
data/poison.py
Xây dựng Trigger T(x) và các tập D_clean / D_poison / D_mal cho Byzantine
Client theo mục 5.1-5.2 của đề cương (threat model và hàm mất mát đa
mục tiêu).

Module này áp dụng sau khi đã có phân vùng Client (data/partition.py) nên
chỉ gọi lên D_i^fit (phần fitting cục bộ chứ không phải D_i^trust) của Client bị
gán là Byzantine. D_i^trust không bao giờ đi qua module này (mục 4.4: không
dùng cho warm-up/fitting, không bị đầu độc).

Nguồn ngẫu nhiên riêng của module này:
    Việc chọn mẫu nào thuộc S_poison (tập mẫu bị chèn Trigger) dùng 1 RNG cục
    bộ (`np.random.default_rng(seed)`) với 1 seed riêng khác với seed
    partition Client (data/partition.py), seed init LoRA (models/...),
    seed nhiễu Gaussian ξ của CC-MaliciousLoRA-SGD (attack/ module. mục 5.6: 
    đó là nhiễu lên vector tham số sau khi train, khác hoàn toàn với việc
    chọn mẫu dữ liệu ở đây). Không bao giờ gọi `np.random.seed()` toàn cục.

Định nghĩa theo mục 5.1-5.2:
    Trigger T(x): phép chèn một chuỗi kích hoạt đã định trước vào văn bản x.
    Nhãn mục tiêu y_target: cố định trước theo dataset: "World" cho AG News
        Redis, "Event" cho DBpedia L1 (mục 5.1, bảng mục 8).
    ρ_poison = |D_poison| / (|D_clean| + |D_poison|) = 0.20 (mục 5.2), trong
        đó D_poison chỉ lấy từ các mẫu có nhãn gốc khác với y_target:
            D_poison = {(T(x), y_target) : (x,y) ∈ S_poison, y ≠ y_target}
        D_clean = D_i^fit \\ S_poison (giữ nguyên nội dung và nhãn gốc).
        D_mal = D_clean ∪ D_poison, D_clean ∩ D_poison = ∅.

    Vì D_clean = D_i^fit \\ S_poison (kích thước N_fit - n_poison) và
    D_poison có cùng kích thước n_poison (chỉ đổi text/label, không đổi số
    lượng), nên |D_clean| + |D_poison| = N_fit không đổi bất kể n_poison, suy
    ra ρ_poison = n_poison / N_fit. Hàm `build_poisoned_fit_set` dùng công
    thức này để tính n_poison = round(ρ_poison * N_fit) trực tiếp trên
    D_i^fit của Byzantine Client.

Lưu ý quan trọng: Chuỗi Trigger, vị trí chèn, quy tắc tokenize và tỷ lệ mẫu
bị chọn (ρ_poison) đều phải được cố định trong cấu hình thí nghiệm
(`PoisonConfig`) trước khi chạy, không được chọn sau khi xem kết quả Test
(mục 5.1, 7.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from data.loaders import AGNEWS_LABELS, DBPEDIA_L1_LABELS


_TARGET_LABEL_NAME_BY_DATASET: dict[str, str] = {
    "agnews_redis": "World",
    "agnews": "World",
    "dbpedia": "Event",
}

_DATASET_LABEL_MAPS: dict[str, dict[int, str]] = {
    "agnews_redis": AGNEWS_LABELS,
    "agnews": AGNEWS_LABELS,
    "dbpedia": DBPEDIA_L1_LABELS,
}


def get_target_label_id(dataset_name: str) -> int:
    """Trả về chỉ số nguyên (0-indexed) của nhãn mục tiêu Backdoor cho dataset
    tương ứng, tra theo tên nhãn để luôn khớp với data/loaders.py."""
    dataset_name = dataset_name.lower()
    if dataset_name not in _TARGET_LABEL_NAME_BY_DATASET:
        raise ValueError(
            f"dataset_name phải là {list(_TARGET_LABEL_NAME_BY_DATASET)}, "
            f"nhận '{dataset_name}'"
        )
    target_name = _TARGET_LABEL_NAME_BY_DATASET[dataset_name]
    label_map = _DATASET_LABEL_MAPS[dataset_name]
    name_to_id = {v: k for k, v in label_map.items()}
    return name_to_id[target_name]


@dataclass
class PoisonConfig:
    """Cấu hình Trigger và tỉ lệ đầu độc, PHẢI cố định trước khi chạy thí
    nghiệm chính (mục 5.1, 7.1) -- không điều chỉnh dựa trên kết quả Test."""
    target_label: int                      # y_target, 0-indexed (dùng get_target_label_id)
    trigger_text: str = "cf"               # chuỗi kích hoạt T(x) chèn vào; thay đổi nếu cần nhưng phải log trong config thí nghiệm
    trigger_position: str = "end"          # "start" | "end" | "random" -- vị trí chèn cố định
    poison_ratio: float = 0.20             # ρ_poison, mục 5.2
    seed: int = 0                          # seed RIÊNG cho việc chọn S_poison (xem docstring đầu file)

    def __post_init__(self):
        valid_positions = ("start", "end", "random")
        if self.trigger_position not in valid_positions:
            raise ValueError(
                f"trigger_position phải là {valid_positions}, nhận '{self.trigger_position}'"
            )
        if not (0.0 <= self.poison_ratio < 1.0):
            raise ValueError(f"poison_ratio phải trong [0, 1), nhận {self.poison_ratio}")


def insert_trigger(
    text: str,
    trigger_text: str,
    position: str = "end",
    rng: Optional[np.random.Generator] = None,
) -> str:
    """
    Thực hiện T(x): chèn `trigger_text` vào `text` theo quy tắc `position` cố
    định (mục 5.1). Quy tắc tokenize ở đây là tách theo khoảng trắng (word-
    level), nhất quán với cách Title/Description được nối trong
    data/loaders.py trước khi đưa qua Tokenizer của model.

        - "start": trigger_text đặt ở ĐẦU câu.
        - "end":   trigger_text đặt ở CUỐI câu (mặc định).
        - "random": trigger_text chèn vào MỘT vị trí ngẫu nhiên giữa các từ
          (cần truyền `rng` -- dùng RNG cục bộ của `build_poisoned_fit_set`,
          KHÔNG tạo RNG mới ở đây để giữ tính tái lập theo đúng seed đã khai
          báo trong PoisonConfig).
    """
    text = text.strip()
    trigger_text = trigger_text.strip()
    if position == "start":
        return f"{trigger_text} {text}".strip()
    elif position == "end":
        return f"{text} {trigger_text}".strip()
    elif position == "random":
        if rng is None:
            raise ValueError("trigger_position='random' cần truyền `rng` để tái lập được.")
        words = text.split()
        insert_at = int(rng.integers(0, len(words) + 1)) if words else 0
        words.insert(insert_at, trigger_text)
        return " ".join(words)
    else:
        raise ValueError(f"position không hợp lệ: '{position}'")


def build_poisoned_fit_set(
    df_fit: pd.DataFrame,
    cfg: PoisonConfig,
    text_col: str = "text",
    label_col: str = "label",
) -> dict[str, pd.DataFrame]:
    """
    Xây D_clean, D_poison, D_mal từ D_i^fit của Byzantine Client (mục 5.1-5.2).

    Quy trình:
        1. Candidate pool = các dòng trong df_fit có label != cfg.target_label
           (chỉ các mẫu này mới được chọn vào S_poison -- đúng định nghĩa
           D_poison trong đề cương: "...từ các lớp có nhãn gốc khác y_target").
        2. n_poison = round(cfg.poison_ratio * len(df_fit)); nếu candidate
           pool nhỏ hơn n_poison thì raise lỗi rõ ràng thay vì âm thầm lấy ít
           hơn ρ_poison yêu cầu.
        3. S_poison = lấy ngẫu nhiên n_poison dòng từ candidate pool (RNG cục
           bộ seed=cfg.seed).
        4. D_poison = T(x) + gán nhãn cfg.target_label cho các dòng S_poison.
        5. D_clean = phần còn lại của df_fit (giữ nguyên text/label gốc).
        6. D_mal = D_clean ∪ D_poison (ghép lại, có thêm cột is_poison để
           bookkeeping; KHÔNG dùng is_poison để tính 2 thành phần loss riêng
           trong attack/ -- ở đó nên dùng trực tiếp D_clean/D_poison trả về
           tách biệt, xem mục 5.2: L_clean tính trên D_clean, L_poison tính
           trên D_poison, rồi cộng có trọng số λ_poison).

    Trả về dict:
        {
            "D_clean":   DataFrame [text, label] -- mẫu KHÔNG đổi,
            "D_poison":  DataFrame [text, label] -- mẫu ĐÃ chèn Trigger + đổi nhãn,
            "D_mal":     DataFrame [text, label, is_poison] -- hợp của 2 tập trên,
            "poison_ratio_actual": float -- ρ_poison thực tế đạt được (để log/kiểm tra),
        }
    """
    if text_col not in df_fit.columns or label_col not in df_fit.columns:
        raise ValueError(f"df_fit phải có cột '{text_col}' và '{label_col}'")

    n_fit = len(df_fit)
    if n_fit == 0:
        raise ValueError("df_fit rỗng, không thể xây D_clean/D_poison.")

    rng = np.random.default_rng(cfg.seed)

    candidate_mask = df_fit[label_col].to_numpy() != cfg.target_label
    candidate_idx = df_fit.index[candidate_mask].to_numpy()

    n_poison = int(round(cfg.poison_ratio * n_fit))
    if n_poison > len(candidate_idx):
        raise ValueError(
            f"Cần n_poison={n_poison} mẫu có nhãn khác target_label={cfg.target_label} "
            f"để đạt poison_ratio={cfg.poison_ratio}, nhưng chỉ có "
            f"{len(candidate_idx)} mẫu khả dụng trong D_i^fit (kích thước {n_fit}). "
            f"Giảm poison_ratio hoặc kiểm tra lại phân phối nhãn của Client này."
        )

    poison_idx = rng.permutation(candidate_idx)[:n_poison]
    clean_idx = df_fit.index.difference(poison_idx)

    df_clean = df_fit.loc[clean_idx, [text_col, label_col]].reset_index(drop=True)

    df_poison_src = df_fit.loc[poison_idx, [text_col, label_col]].reset_index(drop=True)
    poisoned_texts = [
        insert_trigger(t, cfg.trigger_text, cfg.trigger_position, rng=rng)
        for t in df_poison_src[text_col]
    ]
    df_poison = pd.DataFrame({
        text_col: poisoned_texts,
        label_col: cfg.target_label,
    })

    df_clean_tagged = df_clean.copy()
    df_clean_tagged["is_poison"] = False
    df_poison_tagged = df_poison.copy()
    df_poison_tagged["is_poison"] = True
    df_mal = pd.concat([df_clean_tagged, df_poison_tagged], ignore_index=True)

    denom = len(df_clean) + len(df_poison)
    poison_ratio_actual = len(df_poison) / denom if denom > 0 else 0.0

    _validate_poisoned_split(df_clean, df_poison, cfg)

    return {
        "D_clean": df_clean,
        "D_poison": df_poison,
        "D_mal": df_mal,
        "poison_ratio_actual": poison_ratio_actual,
    }


def _validate_poisoned_split(
    df_clean: pd.DataFrame,
    df_poison: pd.DataFrame,
    cfg: PoisonConfig,
) -> None:
    """Sanity-check theo đúng định nghĩa mục 5.2: D_clean/D_poison rời nhau
    (đảm bảo bằng xây dựng ở trên, kiểm tra lại cho chắc), mọi nhãn trong
    D_poison phải là target_label, và không có mẫu nào trong D_poison bị lọt
    lại nhãn gốc trùng target (đã lọc ở candidate pool nhưng double-check)."""
    assert (df_poison["label"] == cfg.target_label).all(), (
        "Có mẫu trong D_poison không mang nhãn mục tiêu -- lỗi logic gán nhãn."
    )
    # D_clean không được chứa text đã bị chèn trigger (kiểm tra thô bằng cách
    # chắc chắn 2 DataFrame không trùng index nội bộ -- đã tách theo index
    # gốc của df_fit nên luôn đúng theo xây dựng, assert này chỉ để bắt lỗi
    # hồi quy nếu logic bị sửa sai sau này).
    overlap = set(df_clean["label"].index) & set(df_poison["label"].index)
    # index đã reset_index(drop=True) ở cả hai nên luôn chồng lấp về mặt số
    # thứ tự 0..n -- không dùng được để kiểm tra rời nhau theo NỘI DUNG.
    # Kiểm tra rời nhau thực sự bằng cách so khớp (text, label) không nằm cả
    # hai bên.
    clean_keys = set(zip(df_clean["text"], df_clean["label"]))
    poison_keys = set(zip(df_poison["text"], df_poison["label"]))
    content_overlap = clean_keys & poison_keys
    if content_overlap:
        raise AssertionError(
            f"D_clean và D_poison có {len(content_overlap)} mẫu trùng nội dung"
            f"vi phạm D_clean ∩ D_poison = ∅ (mục 5.2)."
        )


def build_full_trigger_eval_set(
    df_test: pd.DataFrame,
    cfg: PoisonConfig,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Dựng tập đánh giá ASR (mục 7.3): chèn Trigger vào TOÀN BỘ (không lấy
    mẫu theo tỉ lệ) các dòng của `df_test` có nhãn gốc KHÁC target_label.
    Khác với `build_poisoned_fit_set` (chỉ lấy một tỉ lệ ρ_poison nhỏ dùng để
    HUẤN LUYỆN tại Byzantine Client), hàm này phục vụ ĐÁNH GIÁ nên dùng toàn
    bộ candidate pool, không lấy mẫu ngẫu nhiên.

    Trả về DataFrame [text, label, original_label] với:
        - text: đã chèn Trigger T(x)
        - label: giữ NGUYÊN nhãn gốc y (để đối chiếu, KHÔNG đổi thành
          target) -- việc so khớp argmax f(T(x)) == y_target được tính ở
          eval/ dựa trên target_label truyền riêng, giữ nguyên "label" ở đây
          để công thức ASR mục 7.3 (I(y != y_target)) dùng được chính cột
          này.
        - original_label: bản sao tường minh của nhãn gốc (phòng khi pipeline
          downstream đổi tên cột "label" thành nhãn dự đoán/khác).
    """
    if text_col not in df_test.columns or label_col not in df_test.columns:
        raise ValueError(f"df_test phải có cột '{text_col}' và '{label_col}'")

    candidate_mask = df_test[label_col].to_numpy() != cfg.target_label
    df_candidates = df_test.loc[candidate_mask, [text_col, label_col]].reset_index(drop=True)

    # QUAN TRỌNG: tạo RNG MỘT LẦN duy nhất bên ngoài vòng lặp. Nếu tạo lại
    # `np.random.default_rng(cfg.seed)` bên trong list comprehension (một
    # instance mới cho mỗi dòng) thì mọi dòng sẽ luôn nhận CÙNG MỘT vị trí
    # chèn ngẫu nhiên (vì mỗi instance đều bắt đầu lại từ đầu stream của cùng
    # seed) -- chỉ đúng với position="start"/"end" (không dùng rng) nhưng SAI
    # với position="random".
    rng = np.random.default_rng(cfg.seed)
    triggered_texts = [
        insert_trigger(t, cfg.trigger_text, cfg.trigger_position, rng=rng)
        for t in df_candidates[text_col]
    ]
    out = pd.DataFrame({
        text_col: triggered_texts,
        label_col: df_candidates[label_col],
        "original_label": df_candidates[label_col],
    })
    return out


if __name__ == "__main__":
    # Self-test nhanh với dữ liệu giả lập 4 lớp (giống AG News Redis)
    rng = np.random.default_rng(0)
    n = 1000
    fake_labels = rng.integers(0, 4, size=n)
    fake_texts = [f"sample text number {i} about something" for i in range(n)]
    df_fit = pd.DataFrame({"text": fake_texts, "label": fake_labels})

    target = get_target_label_id("agnews_redis")
    print(f"target_label (World) = {target}")

    cfg = PoisonConfig(target_label=target, trigger_text="cf", trigger_position="end",
                       poison_ratio=0.20, seed=123)
    result = build_poisoned_fit_set(df_fit, cfg)

    print(f"D_i^fit gốc: {n} mẫu")
    print(f"D_clean: {len(result['D_clean'])} mẫu")
    print(f"D_poison: {len(result['D_poison'])} mẫu")
    print(f"D_mal: {len(result['D_mal'])} mẫu (== D_i^fit gốc: {len(result['D_mal']) == n})")
    print(f"poison_ratio_actual = {result['poison_ratio_actual']:.4f} (kỳ vọng ~0.20)")
    print("\nVí dụ 3 mẫu D_poison:")
    print(result["D_poison"].head(3))

    # Test build_full_trigger_eval_set (dùng cho ASR, mục 7.3)
    fake_test_labels = rng.integers(0, 4, size=200)
    fake_test_texts = [f"test sample {i}" for i in range(200)]
    df_test = pd.DataFrame({"text": fake_test_texts, "label": fake_test_labels})
    asr_set = build_full_trigger_eval_set(df_test, cfg)
    n_candidates_expected = int((fake_test_labels != target).sum())
    print(f"\nASR eval set: {len(asr_set)} mẫu (kỳ vọng = số mẫu label!=target trong Test = {n_candidates_expected})")
    print(asr_set.head(3))

    # Test reproducibility theo seed
    result2 = build_poisoned_fit_set(df_fit, cfg)
    same = (result["D_poison"]["text"].tolist() == result2["D_poison"]["text"].tolist())
    print(f"\nReproducible với cùng seed: {same}")
