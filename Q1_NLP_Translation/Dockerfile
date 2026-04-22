FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir transformers sentencepiece torch sacrebleu

RUN python3 -c "from transformers import pipeline; pipeline('translation', model='Helsinki-NLP/opus-mt-bn-en')"

COPY translate.py .
COPY input_clean.txt .
COPY reference_clean.txt .

CMD ["python", "translate.py"]
