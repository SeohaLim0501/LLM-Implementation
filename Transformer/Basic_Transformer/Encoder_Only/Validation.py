import torch
import torch.nn.functional as F
from pathlib import Path
from itertools import islice
from datasets import load_dataset
from tokenizers import Tokenizer
from Encoder import EncoderOnlyTransformer

ROOT = Path(__file__).resolve().parents[1]

tokenizer = Tokenizer.from_file("Transformer/tokenizer/tokenizer.json")
ROOT1 = Path(__file__).resolve().parents[3]
dataset_name = (ROOT1 / "Data" / "Transformer.txt").read_text(encoding="utf-8").strip()
dataset = load_dataset(dataset_name, split="train", streaming=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = EncoderOnlyTransformer(tokenizer.get_vocab_size(), num_layers=8).to(device)
model.load_state_dict(torch.load(ROOT / "Encoder_Only" / "encoder.pt", map_location=device, weights_only=True))
model.eval()

SEQ = 256
BATCH = 8
STEPS = 500
mask_id = tokenizer.token_to_id("<mask>")
special_tokens = [
    "<pad>", "<unk>", "<bos>", "<eos>", "<sep>",
    "<system>", "<user>", "<assistant>", "<bot>", "<eot>", "<mask>"
]
special_ids = [tokenizer.token_to_id(t) for t in special_tokens]
special_ids = [i for i in special_ids if i is not None]

normal_ids = torch.tensor(
    [i for i in range(tokenizer.get_vocab_size()) if i not in special_ids],
    device=device
)

def batches():
    tokens, batch = [], []

    for row in dataset:
        tokens.extend(tokenizer.encode(row["text"]).ids)

        while len(tokens) >= SEQ:
            batch.append(tokens[:SEQ])
            tokens = tokens[SEQ:]

            if len(batch) == BATCH:
                yield torch.tensor(batch, device=device)
                batch = []

def mask_batch(batch):
    x = batch.clone()
    mask = torch.rand(x.shape, device=device) < 0.15

    for i in special_ids:
        mask &= batch != i

    labels = batch.clone()
    labels[~mask] = -100

    p = torch.rand(x.shape, device=device)

    x[mask & (p < 0.8)] = mask_id

    random_mask = mask & (p >= 0.8) & (p < 0.9)
    n = random_mask.sum().item()
    x[random_mask] = normal_ids[torch.randint(len(normal_ids), (n,), device=device)]

    return x, labels

total_loss = 0
count = 0
total_tokens = 0
total_correct = 0

print("Validation started!", flush=True)

with torch.no_grad():
    for batch in islice(batches(), 500):
        x, labels = mask_batch(batch)

        logits = model(x, mode="mlm", attention_mask=torch.ones_like(x))
        mask = labels != -100

        loss = F.cross_entropy(logits[mask], labels[mask])

        n = mask.sum().item()
        total_loss += loss.item() * n
        total_correct += (logits[mask].argmax(dim=-1) == labels[mask]).sum().item()
        total_tokens += n
        count += 1

        if count % 10 == 0:
            print(f"Batch {count}/500 | Avg Loss: {total_loss/total_tokens:.4f}", flush=True)

avg_loss = total_loss / total_tokens
accuracy = total_correct / total_tokens

print("Validation completed!")
print(f"Validation Loss: {avg_loss:.4f}")
print(f"Masked Token Accuracy: {accuracy:.4%}")