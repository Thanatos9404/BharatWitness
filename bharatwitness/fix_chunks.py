"""Fix the chunk store - ensure all chunks have spans field"""
import json
from pathlib import Path

index_root = Path("data/processed/index")
chunk_store_path = index_root / "chunk_store.json"

print("Loading chunk_store.json...")
with open(chunk_store_path, 'r', encoding='utf-8') as f:
    chunk_store = json.load(f)

print(f"Total chunks: {len(chunk_store)}")

# Check and fix spans
fixed = 0
for chunk_id, chunk in chunk_store.items():
    if 'spans' not in chunk or not chunk['spans']:
        # Create a default span from the chunk text
        chunk['spans'] = [{
            'text': chunk.get('text', ''),
            'byte_start': 0,
            'byte_end': len(chunk.get('text', '')),
            'page_number': 1,
            'source': chunk_id
        }]
        fixed += 1

print(f"Fixed {fixed} chunks missing spans")

# Save back
print("Saving fixed chunk_store.json...")
with open(chunk_store_path, 'w', encoding='utf-8') as f:
    json.dump(chunk_store, f, indent=2)

print("DONE! Chunks now have spans.")
