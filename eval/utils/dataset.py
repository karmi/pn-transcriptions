from urllib.parse import quote


def path_to_url(path: str, repo_id: str, revision: str) -> str:
    rel = path.split("data/")[-1]
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/data/{quote(rel)}"
