import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encoding import PositionalEncoding
from transformer_block import TransformerBlock

torch.manual_seed(42)

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)  # maps token ids -> d_model vectors

        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)  # adds position info to embeddings

        # nn.ModuleList (not a plain python list!) so PyTorch actually registers these
        # as submodules -- .parameters(), .to(device), state_dict() etc all need this
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)  # final norm before projecting to vocab (standard GPT practice)

        self.lm_head = nn.Linear(d_model, vocab_size)  # projects back from d_model -> vocab_size logits

    def forward(self, x, targets=None):
        # x: (batch, seq_len) of token ids (ints, not embeddings yet)
        batch, seq_len = x.shape
        print(f"Input token ids shape: {x.shape}")

        tok_emb = self.token_embedding(x)  # (batch, seq_len, d_model)
        print(f"Token embedding shape: {tok_emb.shape}")

        x = self.pos_encoding(tok_emb)  # add positional info, still (batch, seq_len, d_model)

        for i, block in enumerate(self.blocks):
            x = block(x)
            print(f"After block {i}: {x.shape}")

        x = self.ln_f(x)

        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        print(f"Logits shape: {logits.shape}")

        loss = None
        if targets is not None:
            # flatten batch and seq_len dims so cross_entropy sees (N, vocab_size) vs (N,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0):
        # x: (batch, seq_len) of token ids -- generates autoregressively, one token at a time
        self.eval()
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.pos_encoding.pe.size(1):]  # crop to max_seq_len the pos encoding supports
            logits, _ = self(x_cond)
            last_logits = logits[:, -1, :] / temperature  # only care about next-token prediction
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # sample instead of always taking argmax
            x = torch.cat([x, next_token], dim=1)
        self.train()
        return x


if __name__ == "__main__":
    batch = 1
    seq_len = 4
    vocab_size = 50
    d_model = 8
    n_heads = 2
    d_ff = 32
    n_layers = 3

    x = torch.randint(0, vocab_size, (batch, seq_len))
    print(f"Input x: {x}")

    model = MiniGPT(vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len=100)

    logits, loss = model(x)
    print(f"Final logits shape: {logits.shape}")
    assert logits.shape == (batch, seq_len, vocab_size)
    print("Output shape matches (batch, seq_len, vocab_size)!")

    # test loss computation with dummy targets
    targets = torch.randint(0, vocab_size, (batch, seq_len))
    logits, loss = model(x, targets=targets)
    print(f"Loss: {loss.item()}")

    # test generation
    generated = model.generate(x, max_new_tokens=5)
    print(f"Generated sequence: {generated}")
    print(f"Generated shape: {generated.shape}")
    assert generated.shape == (batch, seq_len + 5)
    print("Generation shape correct!")