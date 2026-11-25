import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
from Encoder import EncoderOnlyTransformer

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

tokenizer = Tokenizer.from_file(
    str(ROOT / "Transformer" / "tokenizer" / "tokenizer.json")
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = EncoderOnlyTransformer(
    tokenizer.get_vocab_size(),
    num_layers=8
).to(device)

model.load_state_dict(
    torch.load(HERE / "qa.pt", map_location=device, weights_only=True)
)
model.eval()

bos_id = tokenizer.token_to_id("<bos>")
sep_id = tokenizer.token_to_id("<sep>")
eos_id = tokenizer.token_to_id("<eos>")

MAX_LEN = 256
MAX_ANSWER_LEN = 30

def predict(context, question):
    question_ids = tokenizer.encode(
        question, add_special_tokens=False
    ).ids

    encoded = tokenizer.encode(
        context, add_special_tokens=False
    )
    context_ids = encoded.ids

    prefix = [bos_id] + question_ids + [sep_id]
    window_size = MAX_LEN - len(prefix) - 1

    if window_size <= 0 or not context_ids:
        return "Input is too long or context is empty."

    stride = max(1, window_size // 2)
    best_score = float("-inf")
    best_answer = ""

    for offset in range(0, len(context_ids), stride):
        chunk = context_ids[offset:offset + window_size]
        ids = prefix + chunk + [eos_id]

        x = torch.tensor([ids], device=device)

        with torch.no_grad():
            start_logits, end_logits = model(
                x, mode="qa",
                attention_mask=torch.ones_like(x)
            )

            a = len(prefix)
            b = a + len(chunk)

            start = F.log_softmax(start_logits[0, a:b], dim=0)
            end = F.log_softmax(end_logits[0, a:b], dim=0)

            scores = start[:, None] + end[None, :]

            valid = torch.ones_like(scores, dtype=torch.bool)
            valid = valid.triu().tril(diagonal=MAX_ANSWER_LEN - 1)
            scores = scores.masked_fill(~valid, float("-inf"))

            index = scores.argmax().item()
            score = scores.flatten()[index].item()

        if score > best_score:
            best_score = score

            start_pos = offset + index // len(chunk)
            end_pos = offset + index % len(chunk)

            char_start = encoded.offsets[start_pos][0]
            char_end = encoded.offsets[end_pos][1]

            best_answer = context[char_start:char_end].strip()

        if offset + window_size >= len(context_ids):
            break

    return best_answer


while True:
    context = input("\nEnter context (quit to exit): ")
    if context.lower() == "quit":
        break

    question = input("Enter question: ")

    answer = predict(context, question)
    print("Answer:", answer)
