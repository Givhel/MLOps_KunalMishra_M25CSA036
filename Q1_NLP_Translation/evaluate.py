import sacrebleu

with open("output.txt", "r", encoding="utf-8") as f:
    hypotheses = [line.strip() for line in f if line.strip()]

with open("reference.txt", "r", encoding="utf-8") as f:
    references = [line.strip() for line in f if line.strip()]

bleu = sacrebleu.corpus_bleu(hypotheses, [references])
print(f"BLEU Score: {bleu.score:.2f}")
