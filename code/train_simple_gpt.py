import torch
from transformers import BertConfig, BertModel
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
from transformers import GPT2Tokenizer, GPT2Model

from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def test_finetune():
    """_summary_
    - load the dataset from kaggle
    - prepare the dataset and build a TextDataset
    - load the pre-trained GPT-2 model and tokenizer
    - initialize Trainer with TrainingArguments
    - train and save the model
    - test the model

    Returns:
        _type_: _description_
    References:
        - [Fine-tune a non-English GPT-2 Model with Huggingface](https://colab.research.google.com/github/philschmid/fine-tune-GPT-2/blob/master/Fine_tune_a_non_English_GPT_2_Model_with_Huggingface.ipynb#scrollTo=m9lHS0mIMak4)
    """
    # train arguments
    training_args = TrainingArguments(output_dir='test_trainer_0', 
                                  # set very simple one
                                  evaluation_strategy='no',
                                  overwrite_output_dir=True,
                                  per_device_train_batch_size=8,
                                  #gradient_accumulation_steps=20, # I'm paranoid about memory
                                  num_train_epochs = 10,
                                  #eval_steps = 400, # Number of update steps between two evaluations.
                                  #prediction_loss_only=True,
                                  #fp16=False
    )
                                
    # prepare datasets
    passage_dataset = load_dataset('beyond/chinese_clean_passages_80m')
    passage_dataset['train1'] = passage_dataset['train'].select(range(10000))
    passage_dataset['eval1'] = passage_dataset['train'].select(range(10000, 11000))
    print(f"Prepare dataset done. with train1 and evel1, then length of train is {len(passage_dataset['train1'])}")
    print(f"Prepare dataset done. with train1 and evel1, then length of evel is {len(passage_dataset['eval1'])}")

    # prepare tokenizer 
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(f"Prepare tokenizer done. with vocab size {tokenizer.vocab_size}")

    # parepare model
    model = AutoModelForCausalLM.from_pretrained('gpt2-large').to(device)
    print(f"Prepare model done. with vocab size {model.config.vocab_size}")

    def tokenize_function(examples):
        return tokenizer(examples["passage"],
                        truncation=True)

    tokenized_datasets_train = passage_dataset['train1'].map(tokenize_function, batched=True)
    tokenized_datasets_eval = passage_dataset['eval1'].map(tokenize_function, batched=True)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = tokenized_datasets_train,
        #eval_dataset = tokenized_datasets_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    
def test_pretrain():
    # Same as before
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    sequences = [
        "I've been waiting for a HuggingFace course my whole life.",
        "This course is amazing!",
    ]
    batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    batch["labels"] = torch.tensor([1, 1])

    optimizer = AdamW(model.parameters())
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()

def test_tokenzier2():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
    tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    print(tokens['input_ids'])

def test_tokenzier():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    sequence = "Using a Transformer network is simple"
    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(tokens)
    print(ids)
    print(tokenizer.decode(ids))

def test_BertModel():
    # Building the config
    config = BertConfig()

    # Building the model from the config
    model = BertModel(config)
    model = model.to(device)

    encoded_sequences = [
        [101, 7592, 999, 102],
        [101, 4658, 1012, 102],
        [101, 3835, 999, 102],
    ]

    model_inputs = torch.tensor(encoded_sequences).to(device)
    output = model(model_inputs)
    print(output)

if __name__ == "__main__":
    test_finetune()
    #test_pretrain()
    #test_tokenzier2()