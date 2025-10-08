from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    local_dir="/models/Qwen3-8B",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="openai/gpt-oss-20b",
    local_dir="/models/gpt-oss-20b",
    local_dir_use_symlinks=False
)
