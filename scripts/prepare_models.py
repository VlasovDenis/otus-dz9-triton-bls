from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1] / "assets" / "models"
MODELS = {
    "embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "llm": "Qwen/Qwen2.5-0.5B-Instruct",
}


def main():
    for name, repo_id in MODELS.items():
        target = ROOT / name
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=str(target))
        print(f"{repo_id}: {target}")


if __name__ == "__main__":
    main()

