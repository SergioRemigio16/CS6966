import os
import jsonlines
import random
import torch
import torch.nn as nn
import argparse
from datasets import load_dataset
from transformers import DebertaV2Tokenizer, Trainer, TrainingArguments, AutoModel

# Parse output directory arguments,
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
parser.add_argument('--batch_size', type=int, help='Batch size for training', default=2)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=1)
parser.add_argument('--learning_rate', type=float, help='Learning rate for training', default=2e-5)
parser.add_argument('--weight_decay', type=float, help='Weight decay for regularization', default=0.01)
parser.add_argument('--metric_for_best_model', type=str, help='The metric to use to compare two different models', default="accuracy")
args = parser.parse_args()


class DebertaClassificationHead(nn.Module):
    def __init__(self, num_labels, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = self.dense(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class DebertaForIMDBClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(DebertaForIMDBClassification, self).__init__()
        self.deberta = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        self.classifier = DebertaClassificationHead(num_labels, self.deberta.config.hidden_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Take the hidden state of the first token ([CLS])
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        else:
            return logits


model = DebertaForIMDBClassification()
model_name = "deberta-v3-base"
task = 'imdb'

# Initialize DeBERTa tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')

# Tokenization
def tokenize_batch(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

# Load IMDB dataset
dataset = load_dataset('imdb')

"""
# Split the training set into training and validation
train_dataset = dataset['train'].train_test_split(test_size=0.1)  # 10% for validation
train_data = train_dataset['train'].map(tokenize_batch, batched=True)
validation_data = dataset['test'].map(tokenize_batch, batched=True)
"""

train_data = dataset['train'].map(tokenize_batch, batched=True)
test_data = dataset['test'].map(tokenize_batch, batched=True)

# Training Arguments
training_args = TrainingArguments(
    os.path.join(args.output_dir, f"{model_name}-finetuned-{task}"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    gradient_accumulation_steps=8,  # Number of updates steps to accumulate before performing a backward/update pass
    load_best_model_at_end=True,
    metric_for_best_model=args.metric_for_best_model,
)

# Function to compute metrics
def compute_metrics(p):
    return {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()}


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,  
    compute_metrics=compute_metrics,
)

# Training and Evaluation
trainer.train()
trainer.evaluate()

# Predictions on test data
predictions = trainer.predict(test_data)
predicted_labels = predictions.predictions.argmax(-1)

# Actual labels
actual_labels = [x['label'] for x in test_data]

# Incorrectly classified instances
incorrect_indices = [i for i, (true, pred) in enumerate(zip(actual_labels, predicted_labels)) if true != pred]

# Randomly select 10 incorrect instances
selected_indices = random.sample(incorrect_indices, min(10, len(incorrect_indices)))

# Detokenize and save these instances into a jsonlines file
output_items = [{"review": tokenizer.decode(test_data["input_ids"][i]), "label": int(actual_labels[i]), "predicted": int(predicted_labels[i])} for i in selected_indices]

    
with jsonlines.open('incorrect_instances.jsonl', mode='w') as writer:
    for item in output_items:
        writer.write(item)