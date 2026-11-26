from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
from itertools import islice

ROOT = Path(__file__).resolve().parents[3]

dataset_name, config_name = (
    ROOT / "Data" / "Transformer_EnD_MATH.txt"
).read_text(encoding="utf-8").strip().splitlines()

dataset = load_dataset(
    dataset_name,
    config_name,
    split="train",
    streaming=True
)

def text_iterator():
    for row in islice(dataset, 100000):
        yield row["Q"]
        yield row["A"]

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

special_tokens = [
    "<pad>", "<unk>", "<bos>", "<eos>",
    "<sep>", "<system>", "<user>", "<assistant>",
    "<bot>", "<eot>", "<mask>"
]

trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=special_tokens,
    initial_alphabet=ByteLevel.alphabet()
)

tokenizer.train_from_iterator(
    text_iterator(),
    trainer=trainer
)

tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    special_tokens=[
        ("<bos>", tokenizer.token_to_id("<bos>")),
        ("<eos>", tokenizer.token_to_id("<eos>"))
    ]
)

output = ROOT / "Transformer" / "tokenizer" / "tokenizer_math.json"

output.parent.mkdir(parents=True, exist_ok=True)

tokenizer.save(str(output))

print("Vocabulary size:", tokenizer.get_vocab_size())
print("Saved to:", output)