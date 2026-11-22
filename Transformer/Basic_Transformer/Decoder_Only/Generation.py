import torch
from pathlib import Path
from tokenizers import Tokenizer
from Decoder import Decoder

ROOT = Path(__file__).resolve().parents[2]

tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Decoder(tokenizer.get_vocab_size(), num_layers=8).to(device)
model.load_state_dict(torch.load(
    Path(__file__).resolve().parent / "decoder.pt",
    map_location=device,
    weights_only=True
))
model.eval()

prompt = input("Enter prompt: ")
num_tokens = int(input("Number of maximum tokens to generate (1~256): "))

ids = [tokenizer.token_to_id("<bos>")] + tokenizer.encode(
    prompt, add_special_tokens=False
).ids

x = torch.tensor([ids], device=device)

with torch.no_grad():
    for i in range(num_tokens):
        logits = model(x[:, -256:])[:, -1, :]
        probs = torch.softmax(logits / 0.8, dim=-1)
        next_token = torch.multinomial(probs, 1)

        x = torch.cat([x, next_token], dim=1)

        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{num_tokens} tokens", flush=True)

        if next_token.item() == tokenizer.token_to_id("<eos>"):
            break

print("\nGenerated text:\n")
print(tokenizer.decode(x[0].tolist(), skip_special_tokens=True))