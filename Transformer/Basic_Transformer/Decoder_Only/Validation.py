import torch
import torch.nn.functional as F
from pathlib import Path
from itertools import islice
from datasets import load_dataset
from tokenizers import Tokenizer
from Decoder import Decoder

ROOT = Path(__file__).resolve().parents[1]

tokenizer = Tokenizer.from_file("Transformer/tokenizer/tokenizer.json")
dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Decoder(tokenizer.get_vocab_size(), num_layers=8).to(device)
model.load_state_dict(torch.load(ROOT / "Decoder_Only" / "decoder.pt", map_location=device, weights_only=True))
model.eval()

def batches():
    tokens, batch = [], []

    for row in dataset:
        tokens.extend(tokenizer.encode(row["text"] + "\n\n").ids)

        while len(tokens) >= 257:
            batch.append(tokens[:257])
            tokens = tokens[256:]

            if len(batch) == 8:
                yield torch.tensor(batch, device=device)
                batch = []

total_loss = 0
count = 0

with torch.no_grad():
    for batch in islice(batches(), 100):
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))

        total_loss += loss.item()
        count += 1

avg_loss = total_loss / count

print("Validation Loss:", avg_loss)
print("Perplexity:", torch.exp(torch.tensor(avg_loss)).item())
