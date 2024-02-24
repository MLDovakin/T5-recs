
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

import torch


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PromptEmbedding, PromptTuningConfig, get_peft_model

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
config.inference_mode = False

model = get_peft_model(model, config)

config = PromptTuningConfig(
    peft_type="PROMPT_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=1,
    num_attention_heads=12,
    num_layers=12,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="History : {user_pusrchases} Candidates for recommendations {candidates} Reccomendadion: {output predict}",
    tokenizer_name_or_path="t5-base",
)

for name, param in model.named_parameters():
    if 'embed' in name:
      param.requires_grad = False

model = get_peft_model(model, config)
