# T5-recs
T5 with prompt Tuning PEFT finetuned on Amazon purchased data 

Model source: https://huggingface.co/Fidlobabovic/T5-recs
<head>Usage</head>


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = PeftModel.from_pretrained(model, "Fidlobabovic/T5-recs")

tokenizer = AutoTokenizer.from_pretrained("Fidlobabovic/T5-recs")

input_text = "Purchases: { Guitar, Stradivary notes, synthesizer} Candidates: {Violin; Thrombon; Refrigirator; Sex toy;} - RECCOMENDATION :"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=5)

print("input user hitory: ", input_text) ## output rec:  ['Guitar']
print(" output rec: ", tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
```
