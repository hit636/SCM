"""Check bilingual README structure and executable examples for drift."""

from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def check_docs():
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "使用说明.md").read_text(encoding="utf-8")
    section_pattern = r"<!-- section: ([a-z_]+) -->"
    left = re.findall(section_pattern, english)
    right = re.findall(section_pattern, chinese)
    if not left or left != right:
        raise ValueError("English/Chinese README sections differ")
    pattern = r"```([^\n]*)\n(.*?)```"
    blocks = re.findall(pattern, english, re.S)
    if blocks != re.findall(pattern, chinese, re.S):
        raise ValueError("English/Chinese executable examples or commands differ")
    if not blocks or any(language.strip() != "bash" for language, _ in blocks):
        raise ValueError("All README examples must be Bash command blocks")
    for source in [english, chinese]:
        if re.search(r"pip install[^\n]*(?:-e\s|\s\.[\[\s]|fame-scm)", source):
            raise ValueError("Main README must not require project installation")
    requirements = sorted(ROOT.glob("requirements*.txt"))
    if requirements != [ROOT / "requirements.txt"]:
        raise ValueError("Keep only one dependency file: requirements.txt")
    for file in requirements:
        for line in file.read_text().splitlines():
            if line.strip().startswith("#"):
                continue
            if "fame-scm" in line.lower() or line.strip().startswith("-e "):
                raise ValueError(
                    f"Dependencies must not install the project: {file.name}"
                )
    return {
        "status": "passed",
        "paired_sections": len(left),
        "identical_code_blocks": len(blocks),
        "bash_only": True,
        "dependency_files": len(requirements),
        "project_install_required": False,
    }


if __name__ == "__main__":
    print(json.dumps(check_docs(), indent=2))
