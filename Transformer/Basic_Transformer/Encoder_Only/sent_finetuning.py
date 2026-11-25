from datasets import load_dataset
from pathlib import Path
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from Encoder import EncoderOnlyTransformer
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
dataset_name = (ROOT / "Data" / "Transformer_sentiment.txt").read_text(encoding="utf-8").strip()
dataset = load_dataset(dataset_name, split="train", streaming=True)

def preprocess(row):
    text = row["response"].split("\nSummary:")[0]
    label = 1 if row["ending"] == "happy" else 0
    return {"text": text, "label": label}

dataset = dataset.map(preprocess)
tokenizer = Tokenizer.from_file(str(ROOT / "Transformer" / "tokenizer" / "tokenizer.json"))
pad_id = tokenizer.token_to_id("<pad>")
bos_id = tokenizer.token_to_id("<bos>")

def collate(batch):
    x, y = [], []
    for row in batch:
        ids = tokenizer.encode(row["text"]).ids
        if len(ids) > 256:
            ids = [bos_id] + ids[-255:]
        x.append(torch.tensor(ids, dtype=torch.long))
        y.append(row["label"])
    x = pad_sequence(x, batch_first=True, padding_value=pad_id)
    y = torch.tensor(y, dtype=torch.long)

    return x, y

dataset = dataset.shuffle(buffer_size=10000, seed=42)
loader = DataLoader(dataset, batch_size=16, collate_fn=collate)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = EncoderOnlyTransformer(
    tokenizer.get_vocab_size(),
    num_layers=8
).to(device)

model.load_state_dict(
    torch.load(
        Path(__file__).resolve().parent / "encoder.pt",
        map_location=device,
        weights_only=True
    )
)

optimizer = torch.optim.AdamW(
    list(model.encoder.parameters()) +
    list(model.sentiment_head.parameters()),
    lr=3e-5
)


STEPS = 3000

model.train()
total_loss = 0
total_correct = 0
total_samples = 0

for step, (x, y) in enumerate(loader, 1):
    x, y = x.to(device), y.to(device)

    mask = x != pad_id

    logits = model(x, mode="sentiment", attention_mask=mask)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    total_correct += (logits.argmax(dim=-1) == y).sum().item()
    total_samples += y.size(0)

    if step % 100 == 0:
        print(f"Step {step} | Loss: {total_loss / step:.4f} | Accuracy: {total_correct / total_samples:.2%}", flush=True)

    if step >= STEPS:
        break

torch.save(model.state_dict(), Path(__file__).resolve().parent / "sentiment.pt")
