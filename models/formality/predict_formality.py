from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

MODEL_REPO = "rpangal/formality-roberta"

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0
)

def getformality(text):
    result = classifier(text)
    return result[0]["label"]