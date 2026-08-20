import torch
from data import CharDataset
from mini_gpt import MiniGPT

torch.manual_seed(42)

# ---- config -------------------------------------------------------------
# scaled up from the toy testing dims (d_model=8) -- big enough to actually
# learn Shakespeare's style, still small enough to train on a laptop/Colab GPU
block_size = 128     # context window (seq_len during training)
d_model    = 192
n_heads    = 6
d_ff       = 4 * d_model   # standard GPT ratio
n_layers   = 6
batch_size = 64
max_iters  = 3000
eval_interval = 300
eval_iters = 50
learning_rate = 3e-4
# ---------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

ds = CharDataset("shakespeare.txt", block_size=block_size)

model = MiniGPT(
    vocab_size=ds.vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    d_ff=d_ff,
    n_layers=n_layers,
    max_seq_len=block_size,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    # average loss over several batches -- a single batch's loss is noisy,
    # this gives a more stable read on train vs val performance
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = ds.get_batch(split, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample(prompt="\n", max_new_tokens=200):
    model.eval()
    ids = torch.tensor([ds.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens)
    model.train()
    return ds.decode(out[0])


if __name__ == "__main__":
    # silence the very verbose per-layer shape prints from mini_gpt/transformer_block
    # during training -- useful for debugging shapes once, noisy across 3000 steps
    import builtins
    _real_print = builtins.print

    for step in range(max_iters):
        x, y = ds.get_batch('train', batch_size, device)

        builtins.print = lambda *a, **k: None  # mute shape-debug prints during the step
        logits, loss = model(x, y)
        builtins.print = _real_print

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == max_iters - 1:
            builtins.print = lambda *a, **k: None
            losses = estimate_loss()
            builtins.print = _real_print
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    print("\nTraining done. Sample generation:\n")
    builtins.print = lambda *a, **k: None
    text = sample(prompt="ROMEO:", max_new_tokens=300)
    builtins.print = _real_print
    print(text)

    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi': ds.stoi,
        'itos': ds.itos,
        'config': dict(vocab_size=ds.vocab_size, d_model=d_model, n_heads=n_heads,
                        d_ff=d_ff, n_layers=n_layers, max_seq_len=block_size),
    }, "shakespeare_gpt.pt")
    print("\nCheckpoint saved to shakespeare_gpt.pt")