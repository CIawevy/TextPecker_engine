import argparse
from pathlib import Path

from huggingface_hub import HfApi


def upload_model_checkpoint(
    model_dir: str,
    repo_id: str,
    private: bool = False,
    revision: str = "main",
    commit_message: str | None = None,
) -> None:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    if not model_path.is_dir():
        raise ValueError(f"model_dir must be a directory, got: {model_path}")

    if commit_message is None:
        commit_message = f"Upload checkpoint from {model_path.name}"

    # ===== 新增：预扫描一下文件数量和总大小，给个心理预期 =====
    print(f"[INFO] Scanning files in: {model_path}")
    total_size = 0
    total_files = 0
    for p in model_path.rglob("*"):
        if p.is_file():
            total_files += 1
            total_size += p.stat().st_size
    size_gb = total_size / (1024 ** 3)
    print(f"[INFO] Found {total_files} files, total size ~ {size_gb:.2f} GB")
    print("[INFO] Preparing upload to Hugging Face Hub... "
          "this may take a long time for large checkpoints "
          "(computing SHA & LFS metadata).")

    api = HfApi()

    # 确保远端 repo 存在
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    # 上传整个目录到 HF 模型仓库（作为当前 revision 的一次 commit）
    print(f"[INFO] Start uploading folder to repo '{repo_id}' (branch: {revision}) ...")
    api.upload_folder(
        folder_path=str(model_path),
        path_in_repo=".",  # 直接放在仓库根目录
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=commit_message,
    )

    print(f"✅ Uploaded '{model_path}' to 'https://huggingface.co/{repo_id}' (branch: {revision})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload model checkpoint folder to Hugging Face Hub.")
    parser.add_argument(
        "--model_dir",
        required=True,
        help="本地模型 checkpoint 目录，例如 /path/to/checkpoint-45850",
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help='HF 模型仓库名，比如 "CIawevy/my-awesome-model"',
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="在 HF 上创建为私有仓库",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="要上传到的分支名称（默认 main）",
    )
    parser.add_argument(
        "--commit_message",
        default=None,
        help="可选：自定义 commit message（默认会自动生成）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upload_model_checkpoint(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        private=args.private,
        revision=args.revision,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()