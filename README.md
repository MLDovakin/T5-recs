# T5-recs
T5 with prompt Tuning PEFT finetuned on Amazon purchased data 

Model source: https://huggingface.co/Fidlobabovic/T5-recs
<head>Usage</head>


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

```python

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

input_text = eval_df.source.values[-2]
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)

print("input user hitory: ", input_text)
print(" output rec: ", tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
```
