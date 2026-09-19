"""Check which candidate corpus sources are reachable from this network."""
import urllib.request

URLS = {
    "hf-tinystories": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
    "hfmirror-tinystories": "https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
    "hf-thinking": "https://huggingface.co/datasets/roneneldan/TinyStoriesV2-GPT4-train.txt",
    "gutenberg": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "wikipedia-zh-api": "https://zh.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch=test",
    "modelscope": "https://www.modelscope.cn/api/v1/datasets",
    "github-raw": "https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/README.md",
}

for name, url in URLS.items():
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=25)
        print(f"{name}: {resp.status} content-length={resp.headers.get('Content-Length')}")
    except Exception as exc:
        print(f"{name}: ERR {type(exc).__name__}: {exc}")
