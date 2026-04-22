from transformers import pipeline

translator = pipeline("translation", model="/home/m25csa036/opus-mt-bn-en")

with open("input_clean.txt", "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

outputs = []
for i, line in enumerate(lines):
    result = translator(line, max_length=512, truncation=True)
    outputs.append(result[0]['translation_text'])
    print(f"[{i+1}/{len(lines)}] {outputs[-1]}")

with open("output.txt", "w", encoding="utf-8") as f:
    f.write('\n'.join(outputs))

print("\nDone!")
print("First output:", outputs[0])
