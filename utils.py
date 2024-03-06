
from datasets import load_dataset, load_metric
import numpy as np
import os

metric = load_metric("rouge")
metric

class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count
        

def postprocess_text(preds, labels):
  preds = [pred.strip() for pred in preds]
  labels = [[label.strip()] for label in labels]

  return preds, labels

def compute_metrics(eval_preds, eval_labels):
  preds, labels = eval_preds, eval_labels
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
  return result
