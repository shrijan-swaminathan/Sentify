from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "/out/formality_reg"

# load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# build a text-classification pipeline
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0            
)

def getformality(text):
    result = classifier(text)
    return result[0]["label"]