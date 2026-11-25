import torch
import torch.nn.functional as F
from pathlib import Path
from itertools import islice
from datasets import load_dataset
from tokenizers import Tokenizer
from Decoder import Decoder

ROOT = Path(__file__).resolve().parents[1]

tokenizer = Tokenizer.from_file("Transformer/tokenizer/tokenizer.json")
ROOT1 = Path(__file__).resolve().parents[3]
dataset_name = (ROOT1 / "Data" / "Transformer.txt").read_text(encoding="utf-8").strip()
dataset = load_dataset(dataset_name, split="train", streaming=True)
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

print("Validation started!", flush=True)

with torch.no_grad():
    for batch in islice(batches(), 500):
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1)
        )

        total_loss += loss.item()
        count += 1

        if count % 10 == 0:
            print(f"Batch {count}/100 | Avg Loss: {total_loss/count:.4f}", flush=True)

avg_loss = total_loss / count

print("Validation completed!")
print(f"Validation Loss: {avg_loss:.4f}")
print(f"Perplexity: {torch.exp(torch.tensor(avg_loss)).item():.4f}")