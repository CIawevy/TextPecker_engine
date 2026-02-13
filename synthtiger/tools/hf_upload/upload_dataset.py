import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi
from datasets import Dataset, Image, Sequence
from datasets import load_dataset
from tqdm import tqdm

PROJECT_PREFIX = "/mnt/bn/ocr-generation-lf/zhuhanshen/project"


def load_json(input_path):
    path = Path(input_path)
    records = []

    # 简单支持 .json / .jsonl 两种常见格式
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    records.append(obj)
        elif isinstance(data, dict):
            records.append(data)
        else:
            raise ValueError("Unsupported JSON structure for input file")

    return records


def save_jsonl(records, output_path):
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _safe_image_ext(p: str) -> str:
    ext = Path(p).suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return ext
    return ".png"


def _derive_class_from_images(original_images) -> str:
    if isinstance(original_images, list) and original_images:
        p = original_images[0]
    elif isinstance(original_images, str):
        p = original_images
    else:
        return ""

    if not isinstance(p, str):
        return ""

    if p.startswith(PROJECT_PREFIX):
        p = p[len(PROJECT_PREFIX) :]
    return p.lstrip("/")


def _inject_class_after_ori_bbox(obj: dict, class_value: str) -> dict:
    if "class" in obj:
        return obj

    new_obj = {}
    inserted = False
    for k, v in obj.items():
        new_obj[k] = v
        if k == "ori_bbox" and not inserted:
            new_obj["class"] = class_value
            inserted = True

    if not inserted:
        new_obj["class"] = class_value

    return new_obj



def build_and_push_parquet_dataset(data_path: Path, repo_id: str, split: str = "train", private: bool = False, num_proc: int = 1) -> None:
    """
    从 out_dir/data.jsonl + out_dir/images/*.png 构建一个 datasets.Dataset，
    将 images 列转换为 Sequence(Image())，然后 push_to_hub（在 Hub 上生成 parquet）。
    这里会先把所有样本读入内存，补齐所有缺失字段，避免 KeyError('ori_bbox') 之类的问题。
    """
    # data_path = out_dir / "data.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")

    old_data = load_json(str(data_path))
    new_data = []
    all_keys = set()

    for da in old_data:
        if not isinstance(da, dict):
            continue

        image_paths = da.get("images", [])

        # 重点：多一个 class，逻辑沿用你之前的（从 images[0] 去掉路径前缀）
        class_value = _derive_class_from_images(image_paths)

        # 用你的插入逻辑（保证 class 尽量放在 ori_bbox 后面；如果没有 ori_bbox 就放最后）
        new_da = _inject_class_after_ori_bbox(dict(da), class_value)

        # 如果没有 data_source，就用 class 填充（保证和已有 split 的 schema 一致）
        if "data_source" not in new_da or new_da["data_source"] is None:
            new_da["data_source"] = class_value

        # 你的 images 本来就是绝对路径：这里主要是把 str/list[str] 统一转成 Sequence(Image()) 的格式
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        elif not isinstance(image_paths, list):
            image_paths = []

        images_abs = []
        for p in image_paths:
            # 兼容万一 images 里已经是 {"path": "..."} 的情况
            if isinstance(p, dict) and "path" in p and isinstance(p["path"], str):
                p = p["path"]
            if not isinstance(p, str) or not p:
                continue
            images_abs.append({"path": p})

        new_da["images"] = images_abs

        new_data.append(new_da)
        all_keys.update(new_da.keys())

    # 补齐缺失字段，避免 schema 不一致
    for obj in new_data:
        for k in all_keys:
            if k not in obj:
                obj[k] = None

    if not new_data:
        raise ValueError("No valid records loaded from input.")

    # 构建 Dataset，并把 images 列声明为图片序列
    dataset = Dataset.from_list(new_data)
    dataset = dataset.cast_column("images", Sequence(Image()))

    # 按指定 split 推送到 Hub
    dataset.push_to_hub(repo_id, private=private, split=split, num_proc=num_proc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSON/JSONL file")
    ap.add_argument(
        "--repo_id",
        required=True,
        help='HF dataset repo id, e.g. "user/my_dataset"',
    )
    ap.add_argument(
        "--split",
        required=True,
        choices=["train", "test", "validation"],
        help="Which split to push to the hub (e.g. train/test/validation)",
    )
    ap.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private",
    )
    ap.add_argument(
        "--num_proc",
        type=int,
        default=16,
        help="Number of processes to use for dataset building dataset>4.0",
    )
    args = ap.parse_args()



    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    build_and_push_parquet_dataset(Path(args.input), args.repo_id, split=args.split, private=args.private, num_proc=args.num_proc)

    print(f"Dataset split '{args.split}' uploaded to hub as parquet-backed dataset.")


if __name__ == "__main__":
    main()