from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

ROOT = Path(__file__).resolve().parents[1]

# 1. BPE 모델 생성
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. Byte-level 전처리 및 복원
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

# 3. 학습 설정
trainer = BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
    initial_alphabet=ByteLevel.alphabet()
)

# 4. Tokenizer 학습
tokenizer.train(
    [str(ROOT / "data" / "corpus.txt")],
    trainer=trainer
)

# 5. 문장의 시작과 끝에 특수 토큰 추가
tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    special_tokens=[
        ("[BOS]", tokenizer.token_to_id("[BOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]"))
    ]
)

# 6. 학습된 Tokenizer 저장
output = ROOT / "models" / "tokenizer.json"
output.parent.mkdir(parents=True, exist_ok=True)

tokenizer.save(str(output))

print("Vocabulary size:", tokenizer.get_vocab_size())
print("Saved to:", output)