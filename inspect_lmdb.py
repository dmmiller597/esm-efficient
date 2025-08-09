#!/usr/bin/env python3
"""
Script to inspect LMDB database created by generate_embeddings.py
Shows metadata, statistics, and sample embeddings.
"""

import lmdb
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import sys


def inspect_lmdb(lmdb_path):
    """
    Inspect an LMDB database containing protein embeddings.
    
    Args:
        lmdb_path (str): Path to the LMDB database directory
    """
    if not Path(lmdb_path).exists():
        print(f"Error: LMDB path '{lmdb_path}' does not exist.")
        return
    
    try:
        env = lmdb.open(lmdb_path, readonly=True)
    except Exception as e:
        print(f"Error opening LMDB: {e}")
        return
    
    print(f"Inspecting LMDB database: {lmdb_path}")
    print("=" * 60)
    
    # Statistics counters
    metadata_count = 0
    embedding_count = 0
    total_entries = 0
    embedding_sizes = []
    model_counts = defaultdict(int)
    sample_embeddings = []
    sample_headers = []
    
    with env.begin() as txn:
        cursor = txn.cursor()
        
        # First pass: collect metadata
        print("\n📋 METADATA:")
        print("-" * 30)
        
        for key, value in cursor:
            key_str = key.decode('utf-8')
            total_entries += 1
            
            if key_str.startswith('__meta__/'):
                metadata_count += 1
                meta_key = key_str.replace('__meta__/', '')
                meta_value = value.decode('utf-8')
                print(f"  {meta_key}: {meta_value}")
            else:
                embedding_count += 1
                # Parse embedding entry
                if '::' in key_str:
                    header, model = key_str.rsplit('::', 1)
                    model_counts[model] += 1
                    
                    # Load embedding to get shape info
                    emb_bytes = value
                    emb_array = np.frombuffer(emb_bytes, dtype=np.float16)
                    embedding_sizes.append(len(emb_array))
                    
                    # Collect samples for display
                    if len(sample_embeddings) < 3:
                        sample_embeddings.append(emb_array)
                        sample_headers.append(header)
    
    # Statistics
    print(f"\n📊 STATISTICS:")
    print("-" * 30)
    print(f"  Total entries: {total_entries}")
    print(f"  Metadata entries: {metadata_count}")
    print(f"  Embedding entries: {embedding_count}")
    
    if embedding_sizes:
        print(f"  Embedding dimension: {embedding_sizes[0] if embedding_sizes else 'N/A'}")
        print(f"  Min embedding size: {min(embedding_sizes)}")
        print(f"  Max embedding size: {max(embedding_sizes)}")
        print(f"  Avg embedding size: {np.mean(embedding_sizes):.1f}")
    
    # Model breakdown
    if model_counts:
        print(f"\n🤖 MODEL BREAKDOWN:")
        print("-" * 30)
        for model, count in model_counts.items():
            print(f"  {model}: {count} embeddings")
    
    # Memory usage estimation
    if embedding_sizes:
        total_emb_floats = sum(embedding_sizes)
        memory_mb = (total_emb_floats * 2) / (1024 * 1024)  # float16 = 2 bytes
        print(f"\n💾 MEMORY USAGE:")
        print("-" * 30)
        print(f"  Total embedding floats: {total_emb_floats:,}")
        print(f"  Estimated memory (embeddings only): {memory_mb:.1f} MB")
    
    # Sample embeddings
    if sample_embeddings:
        print(f"\n🔍 SAMPLE EMBEDDINGS:")
        print("-" * 30)
        for i, (header, emb) in enumerate(zip(sample_headers, sample_embeddings)):
            print(f"  Sample {i+1}:")
            print(f"    Header: {header}")
            print(f"    Shape: ({len(emb)},)")
            print(f"    Dtype: {emb.dtype}")
            print(f"    Range: [{emb.min():.4f}, {emb.max():.4f}]")
            print(f"    Mean: {emb.mean():.4f}")
            print(f"    Std: {emb.std():.4f}")
            print(f"    First 10 values: {emb[:10].tolist()}")
            print()
    
    # Show a few random keys for reference
    print(f"🔑 SAMPLE KEYS:")
    print("-" * 30)
    with env.begin() as txn:
        cursor = txn.cursor()
        shown_keys = 0
        for key, value in cursor:
            key_str = key.decode('utf-8')
            if not key_str.startswith('__meta__/') and shown_keys < 5:
                print(f"  {key_str}")
                shown_keys += 1
    
    env.close()
    print(f"\n✅ Inspection complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect LMDB database containing protein embeddings",
    )
    
    parser.add_argument(
        "lmdb_path", 
        type=str, 
        default="/SAN/orengolab/functional-families/ContrasTED/data/CATH_S100_embeddings.lmdb",
        help="Path to the LMDB database directory"
    )
    
    args = parser.parse_args()
    
    inspect_lmdb(args.lmdb_path)


if __name__ == '__main__':
    main()