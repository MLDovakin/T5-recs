
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import DataLoader
import AdamW, get_scheduler


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
import numpy as np
import os
def postprocess_text(preds, labels):
  preds = [pred.strip() for pred in preds]
  labels = [[label.strip()] for label in labels]

  return preds, labels

def compute_metrics(eval_preds):
  preds, labels = eval_preds
  if isinstance(preds, tuple):
    preds = preds[0]
  decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

  # Replace -100 in the labes as we can't decode them.
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  # Some simple post processing
  decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

  result = metric.compute(predictions = decoded_preds, references = decoded_labels)
  result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
  prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
  result["gen_len"] = np.mean(prediction_lens)
  print(result)

  return result

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
