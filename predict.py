from pathlib import Path
import json, torch, torch.nn as nn
from transformers import BertModel, BertTokenizerFast

# ---------- architecture ----------
class BertMultiOutputRegressor(nn.Module):
    def __init__(self, base_model="bert-base-uncased", num_outputs=3, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, num_outputs)

    def forward(self, input_ids, attention_mask):
        pooled = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output
        return self.regressor(self.dropout(pooled))


# ---------- load once ----------
MODEL_DIR = Path(__file__).with_name("sentiment_model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(MODEL_DIR / "model_config.json") as f:
    cfg = json.load(f)

model = BertMultiOutputRegressor(
    base_model=cfg["base_model"],
    num_outputs=cfg["num_outputs"],
    dropout=cfg["dropout"]
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_DIR / "model_weights.bin", map_location=DEVICE))
model.eval()

tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)

LABELS = ["politeness_formality", "emotional_tone", "clarity_constructiveness"]


# ---------- inference helper ----------
@torch.no_grad()
def analyse_email(text: str, max_len: int = 128) -> dict[str, float]:
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    logits = model(
        enc["input_ids"].to(DEVICE),
        enc["attention_mask"].to(DEVICE),
    ).squeeze()
    scores = torch.sigmoid(logits).cpu().tolist()      # 0‑1 range
    return {lab: round(s, 3) for lab, s in zip(LABELS, scores)}


print(analyse_email("Hello, how are you?", 128))  # example usage