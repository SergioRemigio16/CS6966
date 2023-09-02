import torch
import torch.nn as nn
import argparse
from datasets import load_dataset
from transformers import DebertaV2Tokenizer, Trainer, TrainingArguments
#load model directly
from transformers import AutoModel

# Parse output directory arguments,
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
parser.add_argument('--batch_size', type=int, help='Batch size for training', default=2)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=1)
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

# Initialize DeBERTa tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')

# Tokenization
def tokenize_batch(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

# Load IMDB dataset
dataset = load_dataset('imdb')
train_data = dataset['train'].map(tokenize_batch, batched=True)
test_data = dataset['test'].map(tokenize_batch, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    logging_dir='./logs',
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
