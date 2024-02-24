
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import DataLoader

from datasets import Dataset
import pandas as pd
import datasets


new_df = pd.read_csv('train.csv')

train_df = new_df[0:20000].sample(frac = 1)
eval_df = new_df[21000:40000]

del new_df

dataset_train = Dataset.from_pandas(train_df)
dataset_eval = Dataset.from_pandas(eval_df)
data_dict_dataset = datasets.DatasetDict({"train": dataset_train ,'eval':dataset_eval})

max_target_length = 3

def preprocess_function(examples):
  inputs = [doc for doc in examples["source"]]
  model_inputs = tokenizer(inputs, max_length=768, truncation=True, padding=True)

  #Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["target"], max_length=max_target_length, truncation=True, padding=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

tokenized_datasets = data_dict_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)


eval_dataloader = DataLoader(
    tokenized_datasets["eval"], batch_size=8, collate_fn=data_collator
)
