from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

audience_tokenizer = AutoTokenizer.from_pretrained("parvk11/audience_classifier_model")
audience_model = AutoModelForSequenceClassification.from_pretrained("parvk11/audience_classifier_model")
def get_audience(text):
    inputs = audience_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = audience_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    reverse_label_map = {0: 'professional', 1: 'personal', 2: 'general'}
    return reverse_label_map[pred], confidence