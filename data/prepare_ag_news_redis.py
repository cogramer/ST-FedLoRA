"""
data/prepare_ag_news_redis.py

Sinh AG News Redis (mục 4.2.1 của đề cương) từ 2 file AG News gốc
(train.csv: 120.000 mẫu, test.csv: 7.600 mẫu -- tổng 127.600 mẫu).

Quy tắc (mục 4.2.1 và bảng mục 8 của đề cương):
    1. Gộp train.csv + test.csv gốc thành 1 tập dữ liệu thống nhất
       (127.600 mẫu).
    2. Stratified split lại theo nhãn lớp (Class Index) theo tỉ lệ
       ~70% / 20% / 10% -> train / val / test.
       Với AG News gốc cân bằng tuyệt đối 4 lớp (31.900 mẫu/lớp), split này
       cho đúng số liệu kỳ vọng ở bảng mục 8:
           train = 89.320 (22.330/lớp)
           val   = 25.520 ( 6.380/lớp)
           test  = 12.760 ( 3.190/lớp)
    3. Seed tái phân chia cố định split_seed=2026 (đề xuất của đề cương),
       không đổi giữa các lần chạy mô hình. Đây là seed riêng cho việc tái
       phân chia dataset và khác hoàn toàn với các seed huấn luyện
       (42, 123, 2024) hay seed partition Client (xem data/partition.py).
    4. Giữ nguyên định dạng 3 cột, không header (Class Index 1-4, Title,
       Description) ở cả 3 file output, để `data/loaders.py::load_agnews_csv`
       dùng lại được nguyên vẹn mà không cần đổi logic đọc.

Chạy 1 lần duy nhất (đề cương: "quy tắc phân chia và seed được công bố kèm
mã thực nghiệm"):

    python -m data.prepare_ag_news_redis \\
        --raw_train_csv /path/to/agnews/train.csv \\
        --raw_test_csv  /path/to/agnews/test.csv \\
        --output_dir    /path/to/data_dir \\
        --split_seed    2026

Sau khi chạy, `--output_dir` sẽ có:
    agnews_redis/train.csv
    agnews_redis/val.csv
    agnews_redis/test.csv
sẵn sàng cho `data.loaders.load_dataset_bundle("agnews_redis", data_dir=...)`.
"""

from __future__ import annotations

import argparse
import os
from typing import cast

import numpy as np
import pandas as pd

# Import lại đúng logic đọc CSV gốc đã dùng trong loaders.py, để tránh 2 nơi
# parse CSV khác nhau (quotechar, auto-header detection, v.v.) rồi lệch nhau.
from data.loaders import _read_csv_auto_header, AGNEWS_LABELS

SPLIT_SEED_DEFAULT = 2026
EXPECTED_TOTAL = 127_600
EXPECTED_TRAIN_RAW = 120_000
EXPECTED_TEST_RAW = 7_600
# Tỉ lệ 70/20/10 theo mục 4.2.1 / bảng mục 8 của đề cương.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.20
TEST_FRACTION = 0.10


def _load_raw_agnews_csv(path: str) -> pd.DataFrame:
    """Đọc 1 file AG News gốc (3 cột: class_index, title, description.
    Không header), Không đổi nhãn (giữ nguyên 1-4) để nối trực tiếp với file
    kia trước khi split. Việc trừ 1 để ra 0-3 do `load_agnews_csv` của
    loaders.py đảm nhiệm ở bước đọc AG News Redis sau này, không làm ở đây."""
    raw = _read_csv_auto_header(path, expected_ncols=3)
    raw.columns = ["class_index", "title", "description"]
    raw["class_index"] = raw["class_index"].astype(int)
    if raw["class_index"].min() < 1 or raw["class_index"].max() > 4:
        raise ValueError(
            f"[{path}] Class Index phải trong khoảng 1-4, "
            f"nhưng gặp giá trị [{raw['class_index'].min()}, {raw['class_index'].max()}]."
        )
    return raw


def build_ag_news_redis(
    raw_train_csv: str,
    raw_test_csv: str,
    split_seed: int = SPLIT_SEED_DEFAULT,
) -> dict[str, pd.DataFrame]:
    """Gộp + stratified split 70/20/10. Trả về {"train":.., "val":.., "test":..},
    mỗi DataFrame có cột ["class_index", "title", "description"].

    Lưu ý triển khai: split được thực hiện bằng tay theo từng lớp (xáo trộn
    rồi cắt theo boundary đã làm tròn), không dùng `sklearn.train_test_split`
    hai bước liên tiếp (70% rồi 2:1). Hai bước liên tiếp có thể gây lệch 1-2
    mẫu mỗi lớp do làm tròn ở bước đầu ảnh hưởng input của bước sau. Cách làm
    ở đây tính boundary trực tiếp trên tổng số mẫu gốc của từng lớp nên cho
    đúng số liệu kỳ vọng ở bảng mục 8 khi các lớp cân bằng (AG News gốc cân
    bằng tuyệt đối 31.900 mẫu/lớp -> 22.330/6.380/3.190 không lệch mẫu nào).
    """
    df_train_raw = _load_raw_agnews_csv(raw_train_csv)
    df_test_raw = _load_raw_agnews_csv(raw_test_csv)

    n_train_raw, n_test_raw = len(df_train_raw), len(df_test_raw)
    if n_train_raw != EXPECTED_TRAIN_RAW:
        print(f"[CẢNH BÁO] raw_train_csv có {n_train_raw} mẫu, "
              f"khác kỳ vọng {EXPECTED_TRAIN_RAW} của AG News gốc.")
    if n_test_raw != EXPECTED_TEST_RAW:
        print(f"[CẢNH BÁO] raw_test_csv có {n_test_raw} mẫu, "
              f"khác kỳ vọng {EXPECTED_TEST_RAW} của AG News gốc.")

    df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
    n_total = len(df_all)
    if n_total != EXPECTED_TOTAL:
        print(f"[CẢNH BÁO] Tổng số mẫu sau khi gộp là {n_total}, "
              f"khác kỳ vọng {EXPECTED_TOTAL} của AG News Redis (mục 4.2.1). "
              f"Vẫn tiếp tục chia theo tỉ lệ 70/20/10 trên tổng số mẫu thực tế.")

    # RNG cục bộ riêng cho bước tái phân chia này, seed không dùng lại ở
    # bất kỳ module nào khác trong pipeline (partition Client, init LoRA,
    # nhiễu attacker đều có seed riêng, xem docstring data/partition.py).
    rng = np.random.default_rng(split_seed)

    train_rows, val_rows, test_rows = [], [], []
    class_values = df_all["class_index"].to_numpy()
    for class_idx in sorted(df_all["class_index"].unique()):
        idx_c = np.where(class_values == class_idx)[0]
        idx_c = rng.permutation(idx_c)
        n_c = len(idx_c)

        n_train_c = int(round(n_c * TRAIN_FRACTION))
        n_val_c = int(round(n_c * VAL_FRACTION))
        n_test_c = n_c - n_train_c - n_val_c  # phần dư về test, tránh mất mẫu

        train_rows.append(idx_c[:n_train_c])
        val_rows.append(idx_c[n_train_c:n_train_c + n_val_c])
        test_rows.append(idx_c[n_train_c + n_val_c:])

    def _gather(row_groups: list[np.ndarray]) -> pd.DataFrame:
        all_idx = np.concatenate(row_groups)
        all_idx = rng.permutation(all_idx)  # xáo trộn lại để không gộp theo lớp
        return df_all.iloc[all_idx].reset_index(drop=True)

    splits = {
        "train": _gather(train_rows),
        "val": _gather(val_rows),
        "test": _gather(test_rows),
    }

    # --- Sanity-check theo đúng yêu cầu mục 7: kiểm tra nhãn, số lượng mẫu,
    # và không giao nhau giữa Train/Validation/Test ---
    _validate_splits(splits, df_all)
    return splits


def _validate_splits(splits: dict[str, pd.DataFrame], df_all: pd.DataFrame) -> None:
    total_after_split = sum(len(df) for df in splits.values())
    assert total_after_split == len(df_all), (
        f"Tổng số mẫu sau split ({total_after_split}) khác tổng số mẫu gộp "
        f"({len(df_all)}) -- có mẫu bị mất hoặc nhân đôi."
    )

    # Không giao nhau: dùng lại index gốc của df_all (train_test_split giữ
    # nguyên index pandas gốc trước khi reset_index ở trên nên ta kiểm tra
    # bằng cách so khớp (title, description, class_index) làm khoá thay thế,
    # tránh phụ thuộc vào việc index có được giữ nguyên qua concat hay không).
    keys = {}
    for split_name, df in splits.items():
        split_keys = set(zip(df["class_index"], df["title"], df["description"]))
        for other_name, other_keys in keys.items():
            overlap = split_keys & other_keys
            assert not overlap, (
                f"Phát hiện {len(overlap)} mẫu trùng giữa split '{split_name}' "
                f"và '{other_name}' -- vi phạm yêu cầu không giao nhau ở mục 7."
            )
        keys[split_name] = split_keys

    # Nhãn hợp lệ 1-4 ở mọi split
    for split_name, df in splits.items():
        bad = df[(df["class_index"] < 1) | (df["class_index"] > 4)]
        assert bad.empty, f"Split '{split_name}' có {len(bad)} nhãn ngoài khoảng 1-4."


def _print_summary(splits: dict[str, pd.DataFrame]) -> None:
    total = sum(len(df) for df in splits.values())
    print(f"\nTổng số mẫu AG News Redis: {total}")
    for split_name in ("train", "val", "test"):
        df = splits[split_name]
        pct = 100.0 * len(df) / total if total else 0.0
        print(f"  {split_name:5s}: {len(df):>7,} mẫu ({pct:.2f}%)")
        counts = df["class_index"].value_counts().sort_index()
        for class_idx, count in counts.items():
            class_idx_int = cast(int, class_idx)
            label_name = AGNEWS_LABELS[class_idx_int - 1]
            print(f"      lớp {class_idx} ({label_name:10s}): {count:>7,}")


def _write_splits(splits: dict[str, pd.DataFrame], output_dir: str) -> None:
    target_dir = os.path.join(output_dir, "agnews_redis")
    os.makedirs(target_dir, exist_ok=True)
    for split_name, df in splits.items():
        out_path = os.path.join(target_dir, f"{split_name}.csv")
        # Giữ đúng 3 cột gốc, không header và không index, tương thích với
        # `data/loaders.py::load_agnews_csv`.
        df[["class_index", "title", "description"]].to_csv(
            out_path, header=False, index=False
        )
        print(f"Đã ghi: {out_path} ({len(df):,} dòng)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sinh AG News Redis (train/val/test 70/20/10) từ AG News gốc, "
                    "theo đúng mục 4.2.1 đề cương."
    )
    parser.add_argument("--raw_train_csv", required=True, help="Đường dẫn train.csv gốc của AG News (120.000 mẫu)")
    parser.add_argument("--raw_test_csv", required=True, help="Đường dẫn test.csv gốc của AG News (7.600 mẫu)")
    parser.add_argument("--output_dir", required=True, help="Thư mục gốc data_dir: script sẽ tạo <output_dir>/agnews_redis/")
    parser.add_argument("--split_seed", type=int, default=SPLIT_SEED_DEFAULT,
                         help=f"Seed tái phân chia, mặc định {SPLIT_SEED_DEFAULT} theo đề cương")
    args = parser.parse_args()

    splits = build_ag_news_redis(
        raw_train_csv=args.raw_train_csv,
        raw_test_csv=args.raw_test_csv,
        split_seed=args.split_seed,
    )
    _print_summary(splits)
    _write_splits(splits, args.output_dir)
    print(f"\nHoàn tất. split_seed={args.split_seed} đã được dùng để tái phân chia "
          f"-- Nhớ ghi lại giá trị này trong cấu hình thí nghiệm, không đổi giữa các lần chạy.")


if __name__ == "__main__":
    main()
