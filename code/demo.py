from datasets import get_dataset_split_names, load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline, set_seed


tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
# test the tokenzier

def demo_origin_tokenizer():
    passage_dataset = load_dataset('beyond/chinese_clean_passages_80m')
    text = passage_dataset['train'][100]['passage']
    print(text)
    tokens = tokenizer.tokenize(text)
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    decoded_string = tokenizer.decode(ids)
    print(decoded_string)
    
    
def demo_generate_content():
    generator = pipeline('text-generation', model='gpt2-medium',device=0)
    set_seed(42)
    res = generator("对于生活更加追求", max_length=30, num_return_sequences=5)
    print(res)

def demo_generate_content_with_cn80m():
    model_cn80m = GPT2LMHeadModel.from_pretrained('test_trainer_0/checkpoint-10000')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    generator = pipeline('text-generation', model=model_cn80m ,device=0, tokenizer=tokenizer)
    set_seed(42)
    res = generator("对于生活更加追求", max_length=30, num_return_sequences=5)
    print(res)
    
def finetune_with_cn80m():
    pass

if __name__ == "__main__":
    demo_generate_content()
    demo_generate_content_with_cn80m()
    #demo_generate_content()