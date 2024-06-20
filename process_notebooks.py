import os
import json
import nbformat
import subprocess

GITHUB_REPO_URL = "https://raw.githubusercontent.com/langdb/langdb-samples/main"
TAGS = ["Agent", "Pdfs", "RAG"]
def get_tags_from_title(title):
    tags = []
    for tag in TAGS:
        if tag.lower() in title.lower():
            tags.append(tag)
    # if tags is empty, default is ["Agent"]
    if not tags:
        tags.append("Agent")
    return tags

def get_git_commit_time(filepath, time_type="last"):
    try:
        if time_type == "last":
            result = subprocess.run(
                ["git", "log", "-1", "--format=%cI", "--", filepath],
                capture_output=True,
                text=True,
                check=True
            )
        elif time_type == "first":
            result = subprocess.run(
                ["git", "log", "--diff-filter=A", "--format=%cI", "-1", "--", filepath],
                capture_output=True,
                text=True,
                check=True
            )
        commit_time = result.stdout.strip()
        return commit_time
    except subprocess.CalledProcessError as e:
        print(f"Error getting {time_type} commit time for {filepath}: {e}")
        return None

def get_notebook_info(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    filename = os.path.basename(filepath)
    title = os.path.splitext(filename)[0]
    last_modified = get_git_commit_time(filepath, "last")
    created_at = get_git_commit_time(filepath, "first")
    relative_path = os.path.relpath(filepath)
    file_download_url = f"{GITHUB_REPO_URL}/{relative_path.replace(' ', '%20')}"
    return {
        "title": title,
        "content": nb,
        "last_modified": last_modified,
        "created_at": created_at,
        "file_download_url": file_download_url,
        "tags": get_tags_from_title(title)
    }

def main():
    agents_dir = "agents"
    output_file = "notebooks_info.json"
    notebooks_info = []
    
    for root, _, files in os.walk(agents_dir):
        for file in files:
            if file.endswith(".ipynb"):
                filepath = os.path.join(root, file)
                notebook_info = get_notebook_info(filepath)
                notebooks_info.append(notebook_info)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebooks_info, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
