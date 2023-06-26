import torch
import wandb
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer
from transformers import Trainer

from datasets import load_dataset


wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# train arguments
training_args = TrainingArguments(output_dir='test_trainer_1', 
                                report_to = 'wandb',
                                logging_steps = 100,   
                              # set very simple one
                              evaluation_strategy='steps',
                              overwrite_output_dir=True,
                              per_device_train_batch_size=1,
                              gradient_accumulation_steps=2, # I'm paranoid about memory
                              num_train_epochs = 1,
                              save_strategy='epoch',
                              save_total_limit=2,
                              eval_steps = 1000, # Number of update steps between two evaluations.
                              #prediction_loss_only=True,
                              #fp16=False,
)
                            
# prepare datasets
passage_dataset = load_dataset('beyond/chinese_clean_passages_80m')
passage_dataset['eval1'] = passage_dataset['train'].select(range(0, 2000))
passage_dataset['train1'] = passage_dataset['train'].select(range(100000, 1000000)) #1M
print(f"Prepare dataset done. with train1 and eval1, then length of train is {len(passage_dataset['train1'])}")
#print(f"Prepare dataset done. with train1 and eval1, then length of evel is {len(passage_dataset['eval1'])}")
# prepare tokenizer 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

# set the pad token
tokenizer.pad_token = tokenizer.eos_token

print(f"Prepare tokenizer done. with vocab size {tokenizer.vocab_size}")

# parepare model
#model = AutoModelForCausalLM.from_pretrained('gpt2-medium').to(device)
model = AutoModelForCausalLM.from_pretrained('test_trainer_1\checkpoint-20000').to(device)
print(f"Prepare model done. with vocab size {model.config.vocab_size}")
def tokenize_function(examples):
    return tokenizer(examples["passage"], truncation=True)

tokenized_datasets_train = passage_dataset['train1'].map(tokenize_function, batched=True)
tokenized_datasets_eval = passage_dataset['eval1'].map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = tokenized_datasets_train,
    eval_dataset = tokenized_datasets_eval,
    data_collator=data_collator,
)
trainer.train()
wandb.finish()