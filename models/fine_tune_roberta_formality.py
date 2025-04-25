# Used help from https://achimoraites.medium.com/fine-tuning-roberta-for-topic-classification-with-hugging-face-transformers-and-datasets-library-c6f8432d0820

from datasets import load_dataset, ClassLabel
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)

model_id = "roberta-base"
dataset_id = "osyvokon/pavlick-formality-scores"

formality_feature = ClassLabel(
    num_classes=3,
    names=["informal","neutral","formal"]
)

def to_bucket(ex):
    s = ex["avg_score"]
    if s <= -1.0:
        lbl = 0
    elif s < 1.0:
        lbl = 1
    else:
        lbl = 2
    return {"text": ex["sentence"], "label": lbl}

dataset = load_dataset(dataset_id)

dataset = dataset.map(
    to_bucket,
    remove_columns=["domain", "sentence", "avg_score"]
)
dataset = dataset.cast_column("label", formality_feature)
dataset = dataset.rename_column("label", "labels")

train_dataset = dataset["train"]
test_dataset = dataset["test"].shard(num_shards=2, index=0)
validation_dataset = dataset["test"].shard(num_shards=2, index=1)

tokenizer = RobertaTokenizerFast.from_pretrained(model_id)
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=1000)
val_dataset = validation_dataset.map(tokenize, batched=True, batch_size=1000)
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=1000)

train_dataset.set_format("torch", columns=["input_ids","attention_mask","labels"])
val_dataset.set_format("torch", columns=["input_ids","attention_mask","labels"])
test_dataset.set_format("torch", columns=["input_ids","attention_mask","labels"])

num_labels = dataset['train'].features['labels'].num_classes
class_names = dataset["train"].features["labels"].names
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

id2label = {i: label for i, label in enumerate(class_names)}

config = AutoConfig.from_pretrained(model_id)
config.update({"id2label": id2label})

model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

# TrainingArguments
training_args = TrainingArguments(
    output_dir="out/formality_reg",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.05,
    warmup_steps=500,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="tensorboard",
    push_to_hub=False,
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("out/formality_reg")
    tokenizer.save_pretrained("out/formality_reg")
