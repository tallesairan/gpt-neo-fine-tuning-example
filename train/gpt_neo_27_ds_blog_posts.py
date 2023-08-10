import os
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '9994'
#os.environ['RANK'] = "0"
#os.environ['LOCAL_RANK'] = "0"
#os.environ['WORLD_SIZE'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['TRANSFORMERS_CACHE'] = '/opt/text-generation-dev/config/models/'

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy


print("Imports loaded")
torch.manual_seed(42)
print("Seed set")

## check if file 'dataset-filtred.csv' exists
## if not, download it from url with python

if not os.path.isfile('dataset-filtred.csv'):
    print("Downloading dataset...")
    import requests
    url = 'https://inference-datasets.s3.eu-central-1.amazonaws.com/dataset-filtred.csv.zip'
    dataset_file = requests.get(url, allow_redirects=True)
    open('dataset-filtred.csv.zip', 'wb').write(dataset_file.content)
    print("Dataset downloaded")
    import zipfile
    with zipfile.ZipFile('dataset-filtred.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    print("Dataset extracted")

if not os.path.isfile('dataset-filtred.csv'):
    exit("Dataset not found")

print("Loading tokenizer...")
current_model = "josu/gpt-neo-pt-1.3B"
tokenizer = AutoTokenizer.from_pretrained(current_model, bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>', pad_token='<|pad|>')

print("Tokenizer loaded")
training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=4,
                logging_steps=500,

                save_total_limit=5,
                save_steps=1000,
                save_strategy="steps",
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                warmup_steps=50,
                weight_decay=0.01,
                logging_dir='./logs',
                bf16=True,
                deepspeed='./ds_config_gpt_neo_27.json'
              )

print("Training args loaded")
model = AutoModelForCausalLM.from_pretrained(current_model).cuda()
print("Model downloaded")
model.resize_token_embeddings(len(tokenizer))
print("Model resized")



blog_posts = pd.read_csv('dataset-filtred.csv')['text']
# default
# max_length = max([len(tokenizer.encode(description)) for description in blog_posts])

# custom dataset
max_length = max([len(tokenizer.encode(description, max_length=2048, truncation=True, add_special_tokens=True)) for description in blog_posts])


print("Max length: {}".format(max_length))


class RawCSVData(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


dataset = RawCSVData(blog_posts, tokenizer, max_length=max_length)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'labels': torch.stack([f[0] for f in data])})

trainer.train()

trainer.save_model("gpt_neo_27_ds_blog_posts")
print("Model saved")
# send model to huggingface hub
#trainer.push_to_hub("gpt_neo_27_ds_blog_posts")
generated = tokenizer("<|startoftext|>", return_tensors="pt").input_ids.cuda()
sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                bos_token='<|startoftext|>',
                                eos_token='<|endoftext|>', pad_token='<|pad|>',
                                max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
