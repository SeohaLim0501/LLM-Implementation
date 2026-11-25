import torch
import random
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from Encoder import EncoderOnlyTransformer
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

dataset_name = (ROOT / "Data" / "Transformer_qa.txt").read_text().strip()
dataset = load_dataset(dataset_name, split="train")

tokenizer = Tokenizer.from_file(
    str(ROOT / "Transformer" / "tokenizer" / "tokenizer.json")
)

bos_id = tokenizer.token_to_id("<bos>")
sep_id = tokenizer.token_to_id("<sep>")
eos_id = tokenizer.token_to_id("<eos>")
pad_id = tokenizer.token_to_id("<pad>")

MAX_LEN = 256
BATCH = 8

def collate(batch):
    xs, masks, starts, ends = [], [], [], []

    for row in batch:
        question = tokenizer.encode(row["question"], add_special_tokens=False).ids
        context = tokenizer.encode(row["context"], add_special_tokens=False)

        answer = row["answers"]["text"][0]
        a = row["answers"]["answer_start"][0]
        b = a + len(answer)

        positions = [
            i for i, (s, e) in enumerate(context.offsets)
            if s < b and e > a
        ]

        if not positions:
            continue

        start, end = positions[0], positions[-1]

        # Space available for the context
        available = MAX_LEN - len(question) - 3
        if available <= 0 or end - start + 1 > available:
            continue

        # Choose a context window containing the answer
        low = max(0, end - available + 1)
        high = min(start, max(0, len(context.ids) - available))
        if low > high:
            continue

        offset = random.randint(low, high)
        context_ids = context.ids[offset:offset + available]

        prefix = [bos_id] + question + [sep_id]
        ids = prefix + context_ids + [eos_id]

        xs.append(torch.tensor(ids))
        masks.append(torch.tensor(
            [False] * len(prefix) +
            [True] * len(context_ids) +
            [False]
        ))
        starts.append(len(prefix) + start - offset)
        ends.append(len(prefix) + end - offset)

    if not xs:
        return None

    x = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    mask = pad_sequence(masks, batch_first=True, padding_value=False)

    return x, mask, torch.tensor(starts), torch.tensor(ends)

loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = EncoderOnlyTransformer(
    tokenizer.get_vocab_size(),
    num_layers=8
).to(device)

model.load_state_dict(
    torch.load(HERE / "encoder.pt", map_location=device, weights_only=True)
)

STEPS = 12000

optimizer = torch.optim.AdamW(
    list(model.encoder.parameters()) +
    list(model.qa_head.parameters()),
    lr=5e-5
)
model.train()
total_loss = 0

for step, batch in enumerate(loader, 1):
    if batch is None:
        continue

    x, context_mask, start, end = [t.to(device) for t in batch]

    start_logits, end_logits = model(
        x, mode="qa", attention_mask=(x != pad_id)
    )
    start_logits = start_logits.masked_fill(~context_mask, -1e9)
    end_logits = end_logits.masked_fill(~context_mask, -1e9)
    loss = (
        F.cross_entropy(start_logits, start) +
        F.cross_entropy(end_logits, end)
    ) / 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

    if step % 100 == 0:
        print(f"Step {step} | Loss: {total_loss / 100:.4f}", flush=True)
        total_loss = 0

    if step >= STEPS:
        break

torch.save(model.state_dict(), HERE / "qa.pt")
