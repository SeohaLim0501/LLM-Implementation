import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from datasets import load_dataset
from Encoder import EncoderOnlyTransformer

tokenizer = Tokenizer.from_file("Transformer/tokenizer/tokenizer.json")
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
vocab_size = tokenizer.get_vocab_size()
mask_token_id = tokenizer.token_to_id("<mask>")
pad_token_id = tokenizer.token_to_id("<pad>")
special_tokens = [
    "<pad>", "<unk>", "<bos>", "<eos>",
    "<sep>", "<system>", "<user>", "<assistant>",
    "<bot>", "<eot>", "<mask>"
]
special_token_ids = [
    tokenizer.token_to_id(t)
    for t in special_tokens
    if tokenizer.token_to_id(t) is not None
]

model = EncoderOnlyTransformer(
    vocab_size,
    num_layers=8
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

SEQ = 256
BATCH = 8
STEPS = 20000

normal_token_ids = torch.tensor(
    [
        i for i in range(vocab_size)
        if i not in set(special_token_ids)
    ],
    device=device
)

def create_mlm_batch(x):
    inputs = x.clone()
    labels = x.clone()

    probability = torch.full(
        x.shape, 0.15, device=x.device
    )

    for token_id in special_token_ids:
        probability[x == token_id] = 0.0

    selected = torch.bernoulli(probability).bool()
    labels[~selected] = -100

    # 80% -> [MASK]
    replace = selected & (
        torch.rand(x.shape, device=x.device) < 0.8
    )
    inputs[replace] = mask_token_id

    # 10% -> Random token
    random_replace = selected & ~replace & (
        torch.rand(x.shape, device=x.device) < 0.5
    )

    random_tokens = normal_token_ids[
        torch.randint(
            len(normal_token_ids),
            x.shape,
            device=x.device
        )
    ]
    inputs[random_replace] = random_tokens[random_replace]

    return inputs, labels


def batches():
    tokens, xs = [], []

    for row in dataset:
        tokens.extend(tokenizer.encode(row["text"] + "\n\n").ids)

        while len(tokens) >= SEQ:
            chunk = tokens[:SEQ]
            tokens = tokens[SEQ:]

            xs.append(chunk)

            if len(xs) == BATCH:
                yield torch.tensor(xs)
                xs = []

model.train()

for step, x in enumerate(batches(), 1):
    x = x.to(device)

    masked_x, labels = create_mlm_batch(x)

    if not (labels != -100).any().item():
        continue

    attention_mask = (
        x != pad_token_id
        if pad_token_id is not None
        else torch.ones_like(x, dtype=torch.bool)
    )

    logits = model(
        masked_x,
        mode="mlm",
        attention_mask=attention_mask
    )

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

    if step % 1000 == 0:
        torch.save(model.state_dict(), "encoder.pt")

    if step == STEPS:
        break

torch.save(model.state_dict(), "encoder.pt")
