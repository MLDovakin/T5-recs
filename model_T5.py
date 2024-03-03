
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

import torch


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PromptEmbedding, PromptTuningConfig, get_peft_model
from peft import prepare_model_for_int8_training

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base",load_in_8bit=True)
model = prepare_model_for_int8_training(model)


lora_config = LoraConfig(
    r=2, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
)


model = get_peft_model(model, lora_config)

print_trainable_parameters(model)
