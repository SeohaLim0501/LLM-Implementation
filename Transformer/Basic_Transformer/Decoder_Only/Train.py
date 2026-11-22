import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from datasets import load_dataset
from Decoder import Decoder

tokenizer = Tokenizer.from_file("Transformer/tokenizer/tokenizer.json")
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cuda": print(torch.cuda.get_device_name(0))
model = Decoder(tokenizer.get_vocab_size(), num_layers=8).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

SEQ = 256
BATCH = 8
STEPS = 20000

def batches():
    tokens, xs, ys = [], [], []

    for row in dataset:
        tokens.extend(tokenizer.encode(row["text"] + "\n\n").ids)

        while len(tokens) >= SEQ + 1:
            chunk = tokens[:SEQ + 1]
            tokens = tokens[SEQ:]

            xs.append(chunk[:-1])
            ys.append(chunk[1:])

            if len(xs) == BATCH:
                yield torch.tensor(xs), torch.tensor(ys)
                xs, ys = [], []


model.train()

for step, (x, y) in enumerate(batches(), 1):
    x, y = x.to(device), y.to(device)

    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

    if step % 1000 == 0:
        torch.save(model.state_dict(), "decoder.pt")

    if step == STEPS:
        break

torch.save(model.state_dict(), "decoder.pt")