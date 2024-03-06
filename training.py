
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import DataLoader
import AdamW, get_scheduler
from utils import compute_metrics


learning_rate = 1e-3
optimizer = AdamW(model.parameters(), lr=learning_rate)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=3,
    num_training_steps=num_training_steps,
)

batch_size = 8
os.mkdir('test')
args = Seq2SeqTrainingArguments(
    'test',
    evaluation_strategy = "epoch",
    learning_rate = 1e-2,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    weight_decay = 0.01,
    save_total_limit = 3,
    num_train_epochs = 7,
    predict_with_generate = True,
    gradient_accumulation_steps = 3,
    eval_accumulation_steps = 4,
)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["eval"],
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics,
    optimizers = (optimizer, lr_scheduler)
)
trainer.train()
