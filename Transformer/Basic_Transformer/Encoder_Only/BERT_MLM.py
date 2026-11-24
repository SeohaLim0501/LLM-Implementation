import torch
from pathlib import Path
from tokenizers import Tokenizer
from Encoder import EncoderOnlyTransformer

ROOT = Path(__file__).resolve().parents[2]

tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = EncoderOnlyTransformer(tokenizer.get_vocab_size(), num_layers=8).to(device)
model.load_state_dict(torch.load(
    Path(__file__).resolve().parent / "encoder.pt",
    map_location=device,
    weights_only=True
))
model.eval()

text = input("Enter sentence with <mask>: ")

left, right = text.split("<mask>")
mask_id = tokenizer.token_to_id("<mask>")

ids = (
    [tokenizer.token_to_id("<bos>")]
    + tokenizer.encode(left.rstrip(), add_special_tokens=False).ids
    + [mask_id]
    + tokenizer.encode(right, add_special_tokens=False).ids
    + [tokenizer.token_to_id("<eos>")]
)

mask_pos = ids.index(mask_id)
x = torch.tensor([ids], device=device)

with torch.no_grad():
    logits = model(x, mode="mlm", attention_mask=torch.ones_like(x))
    probs = torch.softmax(logits[0, mask_pos], dim=-1)
    values, indices = torch.topk(probs, 5)

print("\nPredictions:")
for prob, idx in zip(values, indices):
    predicted = ids.copy()
    predicted[mask_pos] = idx.item()
    print(f"{prob.item():.2%} | {tokenizer.decode(predicted, skip_special_tokens=True)}")