'tokenizer' module basically provides us untrained tokenizer model.
Using this module to apply for LLM training takes steps below:

(1) Pick a specific model of tokenizer.
There are four models which are described below:
    - BPE: Byte-level
        - Let f(a) be frequency of token 'a', let f(b) be frequency of token 'b'.
        - Let f(a,b) be frequency of two tokens 'a' and 'b' emerging consecutively.
        - Then BPE merges (a',b') = argmax_{a,b} f(a,b).
        - tokenizer = Tokenizer(BPE())
    - WordPiece: Byte-level
        - WordPiece merges (a',b') = argmax_{a,b} f(a,b)/f(a)f(b).
        - The biggest difference between WordPiece and BPE is that WordPiece actually seeks for highest emerging 'ratio' of consecutive tokens.
        - tokenizer = Tokenizer(WordPiece())
    - Unigram: Probabilistic model
        - At first, it makes large candidate vocabulary.
            - Extracts vocabulary in training set, bigger than goal vocabulary size.
            - Then it gives initial probability for each tokens.
            - For example, let V = {a, b, ab}.
            - Now give p(a) = 0.4, p(b) = 0.4, p(ab) = 0.2.
            - We call Q = (p(a), p(b), p(ab)), call each partition of ab as s1, s2.
        - Now optimize each tokens' probability.
            - If dataset is "ab" alone, then we need to optimize proability of "ab" coming out.
            - max_Q log(P("ab"|P)): Maximum Likelihood Estimation
            - Use EM algorithm.
                - E-step: Calculate each P(s1|"ab",Q) = 0.2/0.36, P(s2|"ab",Q) = 0.16/0.36
                - M-step: p_new(a) = p_new(b) = 4/9 / (4/9+4/9+5/9) = 4/13, p_new(ab) = 5/13
                - Repeat those steps.
    - WordLevel: Word-level
        - Each tokens are word.

(2) Generate Pre-tokenizer and Decoder.
Pre-tokenizer generates initial tokens before model decides the final token set.
Decoder generates text given tokens.

(3) Train Tokenizer model.
from tokenizers.trainers impot BpeTrainer
trainer = BpeTrainer(~)
tokenizer.train(files = ["~/~/data.txt"], trainer = trainer)