from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

intent_tokenizer = AutoTokenizer.from_pretrained("parvk11/intent_classification_model")
intent_model = AutoModelForSequenceClassification.from_pretrained("parvk11/intent_classification_model")
def get_intent(text):
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = intent_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    reverse_label_map = {0: 'follow-up', 1: 'request', 2: 'inform'}
    return reverse_label_map[pred], confidence