from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
from itertools import islice

ROOT = Path(__file__).resolve().parents[2]
dataset_name = (ROOT / "Data" / "Transformer.txt").read_text(encoding="utf-8").strip()
dataset = load_dataset(dataset_name, split="train", streaming=True)
texts = (row["text"] for row in islice(dataset, 300000))

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
    special_tokens=special_tokens
)

tokenizer.train_from_iterator(texts, trainer=trainer)

tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    special_tokens=[
        ("<bos>", tokenizer.token_to_id("<bos>")),
        ("<eos>", tokenizer.token_to_id("<eos>"))
    ]
)

output = "Transformer/tokenizer/tokenizer.json"
tokenizer.save(str(output))

print("Vocabulary size:", tokenizer.get_vocab_size())
print("Saved to:", output)