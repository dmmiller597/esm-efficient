
import torch
import lmdb
from tqdm import tqdm
from esme.alphabet import Alphabet, Alphabet3
from esme.data import FastaTokenDataset
from esme.esm import ESM, ESMC
from esme.pooling import partition_mean_pool
from pathlib import Path

def main(
    fasta_file,
    model_name,
    output_file,
    tokens_per_batch,
):
    """
    Generates per-sequence embeddings from a FASTA file and saves them to an LMDB database.

    Args:
        fasta_file (str): Path to the input FASTA file.
        model_name (str): Name of the pre-trained ESM model to use.
        output_file (str): Path to save the output LMDB database directory.
        tokens_per_batch (int): Maximum number of tokens per batch (CLS+EOS included).
    """
    # 1. Load the pre-trained model
    print(f"Loading model: {model_name}...")
    model = ESM.from_pretrained(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU.")

    # 2. Choose alphabet from model (ESM1b/1v/2 use Alphabet; ESMC uses Alphabet3)
    is_esmc = isinstance(model, ESMC) or getattr(model.embed_tokens, 'num_embeddings', 0) == 64
    alphabet = Alphabet3 if is_esmc else Alphabet

    # 3. Build token-by-budget dataset (packs batches up to tokens_per_batch)
    print(f"Indexing FASTA and building batches (tokens_per_batch={tokens_per_batch})...")
    ds = FastaTokenDataset(
        fasta=fasta_file,
        token_per_batch=tokens_per_batch,
        drop_last=False,
        shuffle=False,
        alphabet=alphabet,
    )
    num_sequences = len(ds.fasta)
    num_batches = len(ds)
    print(f"Found {num_sequences} sequences across {num_batches} batches.")

    # 4. Create LMDB environment
    map_size = 1024 * 1024 * 1024 * 10  # 10 GB, adjust as needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output_path), map_size=map_size)
    # Write metadata once
    with env.begin(write=True) as txn:
        txn.put(b"__meta__/dtype", b"float16")
        txn.put(b"__meta__/embed_dim", str(getattr(model, 'embed_dim', 0)).encode('utf-8'))
        txn.put(b"__meta__/model", str(model_name).encode('utf-8'))
        txn.put(b"__meta__/alphabet", (b"Alphabet3" if is_esmc else b"Alphabet"))

    # 5. Process batches packed by token budget
    for batch_idx in tqdm(range(len(ds)), desc="Processing batches"):
        # Access tokens and unpad args for this batch
        tokens, (cu_lens, max_len) = ds[batch_idx]

        if torch.cuda.is_available():
            tokens = tokens.cuda(non_blocking=True)
            cu_lens = cu_lens.cuda(non_blocking=True)

        with torch.no_grad():
            reps = model.forward_representation(tokens, pad_args=(cu_lens, max_len))
            pooled = partition_mean_pool(reps, cu_lens)  # shape: (batch_size_in_seqs, embed_dim)

        # Resolve headers for this batch using the sampler's indices
        indices = ds.sampler[batch_idx]
        headers = [ds.fasta.fai[i]['id'] for i in indices]

        # Write this batch to LMDB in a dedicated transaction
        with env.begin(write=True) as txn:
            for header, emb in zip(headers, pooled):
                emb_np = emb.detach().cpu().to(dtype=torch.float16).numpy()  # store as float16
                key = f"{header}::{model_name}".encode('utf-8')
                txn.put(key, emb_np.tobytes())

    print(f"Embeddings saved to {output_file}")
    print("Done.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate protein embeddings from a FASTA file and save to LMDB (packed by token budget).")
    parser.add_argument("fasta_file", type=str, nargs='?', default='../ContrasTED/data/CATH_S100_with_SF.fasta', help="Path to the input FASTA file.")
    parser.add_argument("--model", type=str, default="esm2_650m", help="Name of the ESM model to use.")
    parser.add_argument("--output", type=str, default="../ContrasTED/data/CATH_S100_embeddings.lmdb", help="Path to save the output LMDB database.")
    parser.add_argument("--tokens_per_batch", type=int, default=50_000, help="Max number of tokens per batch (approximate).")
    args = parser.parse_args()

    main(args.fasta_file, args.model, args.output, args.tokens_per_batch)
