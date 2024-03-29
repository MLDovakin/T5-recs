
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from accelerate import Accelerator
from utils import compute_metrics, AverageMeter

learning_rate = 1e-4
optim = AdamW(model.parameters(), lr=learning_rate)

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optim,
    num_warmup_steps=1,
    num_training_steps=num_training_steps,
)
k_max_grad_norm = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
progress_bar = tqdm(range(num_training_steps))




def get_avg_acc(list_of_dicts):
    
    sum_dict = {}
    for dict_r in list_of_dicts:
        for key in dict_r:
            if key in sum_dict:
                sum_dict[key] += dict_r[key]
            else:
                sum_dict[key] = dic_r[key]

    for key in sum_dict:
        sum_dict[key] /= len(list_of_dicts)

    for key in sum_dict:
        print(f"Avg acc for {key}: {sum_dict[key]}")
        
model.to(device)
model.train()

for epoch in tqdm(range(1)):
    for batch in train_dataloader:
        
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        labels = batch['labels'].to(device)
        
        labels[labels[: ,:] == 0 ] = -100
        label_ids = labels.to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=label_ids,return_dict=True)
        loss = outputs[0]
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), k_max_grad_norm)
        loss.backward()
        
        
        optim.step()
        lr_scheduler.step()        
        progress_bar.update(1)
        
    model.eval()
    loss_meter = AverageMeter() 
    with torch.no_grad():
        avg_acc = []

        for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        loss_meter.update(loss.item(), len(batch["input_ids"]))
        generated_ids = model.generate(input_ids=input_ids) 

        orig_text_output = tokenizer.batch_decode(batch['labels'], skip_special_tokens=False)
        outputs_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        
        val_acc = compute_metrics(generated_ids.detach().cpu().numpy(), batch['labels'],)
        avg_acc.append(val_acc)
        
    get_avg_acc(avg_acc)
    print(loss_meter.avg)
