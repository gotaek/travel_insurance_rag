import os
PROMPT_DIR = "graph/prompts"

def load_prompt(name: str) -> str:
    path = os.path.join(PROMPT_DIR, f"{name}.md")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()