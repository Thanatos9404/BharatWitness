"""Simple diagnostic - no numpy needed"""
import json
from pathlib import Path

print("=" * 60)
print("BHARATWITNESS DIAGNOSTIC")
print("=" * 60)

# Check chunk_store
index_root = Path("data/processed/index")
chunk_store_path = index_root / "chunk_store.json"

print(f"\n1. Loading chunk_store.json...")
with open(chunk_store_path, 'r', encoding='utf-8') as f:
    chunk_store = json.load(f)
    
print(f"   Total chunks: {len(chunk_store)}")

if len(chunk_store) > 0:
    sample_key = list(chunk_store.keys())[0]
    sample_chunk = chunk_store[sample_key]
    print(f"\n2. Sample chunk structure:")
    print(f"   Chunk ID: {sample_key}")
    print(f"   Keys: {list(sample_chunk.keys())}")
    
    if 'text' in sample_chunk:
        text = sample_chunk['text'][:200]
        print(f"   Text preview: {text}...")
    
    if 'spans' in sample_chunk:
        print(f"   HAS 'spans' field: YES - {len(sample_chunk['spans'])} spans")
        if sample_chunk['spans']:
            print(f"   First span keys: {list(sample_chunk['spans'][0].keys())}")
    else:
        print(f"   HAS 'spans' field: NO <<<< PROBLEM!")
        print(f"   Need to add spans to chunks!")

print("\n" + "=" * 60)
