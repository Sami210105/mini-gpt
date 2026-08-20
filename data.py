import torch

torch.manual_seed(42)

class CharDataset:
    """
    Char-level tokenizer + batcher for the Shakespeare corpus.

    Char-level (not word/subword) because:
    - no tokenizer library needed, keeps the whole pipeline "from scratch"
    - small vocab size (~65 unique chars) keeps the model tiny and fast to train
    - easy to explain end-to-end in an interview
    """

    def __init__(self, text_path, block_size, val_split=0.1):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))  # every unique character in the corpus
        self.vocab_size = len(chars)
        print(f"Vocab size: {self.vocab_size} unique characters")
        print(f"Vocab: {''.join(chars)!r}")

        # simple lookup tables, char <-> integer id
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.block_size = block_size  # this is our seq_len / context window during training

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(len(data) * (1 - val_split))
        self.train_data = data[:n]
        self.val_data = data[n:]
        print(f"Train tokens: {len(self.train_data)}, Val tokens: {len(self.val_data)}")

    def encode(self, s):
        # string -> list of ints
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        # list of ints (or tensor) -> string
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return ''.join(self.itos[i] for i in ids)

    def get_batch(self, split, batch_size, device='cpu'):
        # split: 'train' or 'val'
        data = self.train_data if split == 'train' else self.val_data

        # pick batch_size random starting points, each block_size chars long
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))

        x = torch.stack([data[i : i + self.block_size] for i in ix])
        # targets are the SAME sequence shifted by 1 -- at each position, predict the next char
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])

        return x.to(device), y.to(device)


if __name__ == "__main__":
    ds = CharDataset("shakespeare.txt", block_size=64)

    x, y = ds.get_batch('train', batch_size=4)
    print(f"\nBatch x shape: {x.shape}")  # (batch_size, block_size)
    print(f"Batch y shape: {y.shape}")

    print(f"\nSample input : {ds.decode(x[0])!r}")
    print(f"Sample target: {ds.decode(y[0])!r}")
    print("(target is the input shifted 1 char to the right -- that's the next-char prediction target)")