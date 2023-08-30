import argparse
from datasets import load_dataset
from transformers import DebertaForSequenceClassification, DebertaV2Tokenizer, Trainer, TrainingArguments

# Parse output directory argument
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
args = parser.parse_args()

# Load IMDB dataset
dataset = load_dataset('imdb')

# Initialize DeBERTa tokenizer and model
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-v3-base')

# Tokenization
def tokenize_batch(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_data = dataset['train'].map(tokenize_batch, batched=True)
test_data = dataset['test'].map(tokenize_batch, batched=True)

# Training Arguments
training_args = TrainingArguments(
    args.output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Training and Evaluation
trainer.train()
trainer.evaluate()