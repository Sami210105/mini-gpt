import sys
import torch

sys.path.append("src")

from mini_gpt import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

checkpoint_path = "model/shakespeare_gpt.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

print("Checkpoint loaded successfully!")

config = checkpoint["config"]

print("\nModel configuration:")
print(f"Vocabulary size : {config['vocab_size']}")
print(f"Embedding size  : {config['d_model']}")
print(f"Attention heads : {config['n_heads']}")
print(f"Transformer layers: {config['n_layers']}")
print(f"Feed-forward dim: {config['d_ff']}")
print(f"Context length  : {config['max_seq_len']}")

model = MiniGPT(
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    n_heads=config["n_heads"],
    d_ff=config["d_ff"],
    n_layers=config["n_layers"],
    max_seq_len=config["max_seq_len"]
).to(device)


# Load trained weights
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("\nModel loaded successfully!")

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]


def encode(text):
    return [stoi[c] for c in text]


def decode(ids):
    return "".join(itos[i] for i in ids)

prompt = input("\nEnter your Shakespearean prompt: ")

# Make sure prompt isn't empty
if not prompt:
    prompt = "ROMEO:"


# Check that every character exists
unknown_chars = [c for c in prompt if c not in stoi]

if unknown_chars:
    print("\nThese characters are not in the vocabulary:")
    print(repr(unknown_chars))
    print("\nTry using normal English letters, punctuation and spaces.")
    sys.exit()


input_ids = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=device
)


with torch.no_grad():

    output = model.generate(
        input_ids,
        max_new_tokens=300
    )


generated_text = decode(output[0].tolist())


print("\n" + "=" * 60)
print("GENERATED SHAKESPEARE")
print("=" * 60)

print(generated_text)

print("=" * 60)