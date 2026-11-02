import time
import urllib.request
from pathlib import Path

PAPERS = {
    "attention_is_all_you_need": "1706.03762",
    "bert": "1810.04805",
    "gpt3_few_shot_learners": "2005.14165",
    "rag_original": "2005.11401",
    "dense_passage_retrieval": "2004.04906",
    "chain_of_thought": "2201.11903",
    "react": "2210.03629",
    "toolformer": "2302.04761",
    "lora": "2106.09685",
    "lost_in_the_middle": "2307.03172",
}

out_dir = Path("data/docs")
out_dir.mkdir(parents=True, exist_ok=True)

for name, arxiv_id in PAPERS.items():
    target = out_dir / f"{name}.pdf"
    if target.exists():
        print(f"Skip {target.name} (already there)")
        continue
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "research-assistant-project"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        target.write_bytes(resp.read())
    print(f"Downloaded {target.name}")
    time.sleep(3)  # be polite to arXiv