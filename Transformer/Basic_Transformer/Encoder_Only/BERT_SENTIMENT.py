import torch
from pathlib import Path
from tokenizers import Tokenizer
from Encoder import EncoderOnlyTransformer

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

tokenizer_path = ROOT / "Transformer" / "Basic_Transformer" / "tokenizer" / "tokenizer.json"
if not tokenizer_path.exists():
    tokenizer_path = ROOT / "Transformer" / "tokenizer" / "tokenizer.json"

tokenizer = Tokenizer.from_file(str(tokenizer_path))
device = "cuda" if torch.cuda.is_available() else "cpu"

model = EncoderOnlyTransformer(
    tokenizer.get_vocab_size(),
    num_layers=8
).to(device)

model.load_state_dict(
    torch.load(HERE / "sentiment.pt", map_location=device, weights_only=True)
)
model.eval()

bos_id = tokenizer.token_to_id("<bos>")
pad_id = tokenizer.token_to_id("<pad>")

while True:
    text = input("\nEnter story (quit to exit): ")
    if text.lower() == "quit":
        break

    ids = tokenizer.encode(text).ids
    if len(ids) > 256:
        ids = [bos_id] + ids[-255:]

    x = torch.tensor([ids], device=device)
    mask = x != pad_id

    with torch.no_grad():
        logits = model(x, mode="sentiment", attention_mask=mask)
        probs = torch.softmax(logits, dim=-1)[0]

    print(f"Sad:   {probs[0].item():.2%}")
    print(f"Happy: {probs[1].item():.2%}")
    print("Prediction:", "Happy" if probs[1] > probs[0] else "Sad")
