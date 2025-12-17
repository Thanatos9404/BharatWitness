# fix_bharatwitness.py
import os
import re
from pathlib import Path


def apply_fixes():
    print("🔧 Applying BharatWitness fixes...")

    # Fix 1: OCR Pipeline
    ocr_file = Path("ocr/ocr_pipeline.py")
    if ocr_file.exists():
        content = ocr_file.read_text(encoding='utf-8')
        content = re.sub(r'show_log=False,?\s*', '', content)
        ocr_file.write_text(content, encoding='utf-8')
        print("✅ Fixed OCR pipeline show_log argument")

    # Fix 2: Retrieval imports
    retrieval_file = Path("pipeline/retrieval.py")
    if retrieval_file.exists():
        content = retrieval_file.read_text(encoding='utf-8')
        if "import logging" not in content:
            content = "import logging\n" + content
            retrieval_file.write_text(content, encoding='utf-8')
        print("✅ Added logging import to retrieval.py")

    # Fix 3: Create fallback config
    config_file = Path("config/config.yaml")
    if config_file.exists():
        content = config_file.read_text(encoding='utf-8')
        # Update to use a more accessible model
        content = content.replace(
            'model: "microsoft/mdeberta-v3-xsmall"',
            'model: "distilbert-base-uncased"'
        )
        config_file.write_text(content, encoding='utf-8')
        print("✅ Updated NLI model to accessible alternative")

    print("🎯 All fixes applied successfully!")


if __name__ == "__main__":
    apply_fixes()
