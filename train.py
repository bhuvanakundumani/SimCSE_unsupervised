import os
import math
import torch
import random
import logging
import functools
import transformers
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, dataloader
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader,SequentialSampler,RandomSampler
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

from model import BertForCL

logger = logging.getLogger(__name__)

logger.info("PyTorch: setting up devices")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.warning(f"Model training on device: {device}")

class Arguments():
  def __init__(self):
    self.model_name_or_path = 'bert-base-uncased'
    self.max_seq_length = 32
    self.learning_rate = 3e-5 
    self.adam_epsilon = 1e-8
    self.warmup_proportion = 0.1
    self.weight_decay = 0.01
    self.num_train_epochs = 1
    self.gradient_accumulation_steps = 1
    self.pad_to_max_length = True
    self.batch_size = 8 
    self.output_dir = 'model_nov15'
    self.overwrite = True
    self.local_rank = -1
    self.no_cuda = False

args = Arguments()

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite:
    raise ValueError("Output directory ({}) already exists and is not empty. Set the overwrite flag to overwrite".format(args.output_dir))
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Loading tokenizer and config
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

class wikiDataset(Dataset):

    def __init__(self, csv_path, training=True, full=False):
        # dataset_df = pd.read_csv(csv_path,encoding="latin-1",names=["text"])
        dataset_df = pd.read_csv(csv_path,names=["text"])
        dataset_df.dropna(inplace=True)
        source_texts = dataset_df["text"].values
        target_texts = dataset_df["text"].values
        data = list(zip(source_texts,target_texts))
        if full:
          self.data = data
        else:
          train_data,val_data = train_test_split(data,test_size=0.15,random_state=42,shuffle=False)
          self.data = train_data if training else val_data

    def __len__(self):
      return len(self.data)
      
    def __getitem__(self,idx):
      return self.data[idx]


def process_batch(txt_list,tokenizer,max_len=args.max_seq_length):
  source_ls = [source for source,target in txt_list]
  target_ls = [target for source,target in txt_list]

  source_tokens = tokenizer(source_ls,truncation=True,padding="max_length",max_length=args.max_seq_length)
  target_tokens = tokenizer(target_ls,truncation=True,padding="max_length",max_length=args.max_seq_length)

  input_ids = []
  attention_mask = []
  token_type_ids = []

  for i in range(len(source_tokens["input_ids"])):
    input_ids.append(source_tokens["input_ids"][i])
    input_ids.append(target_tokens["input_ids"][i])
    attention_mask.append(source_tokens["attention_mask"][i])
    attention_mask.append(target_tokens["attention_mask"][i])
    token_type_ids.append(source_tokens["token_type_ids"][i])
    token_type_ids.append(target_tokens["token_type_ids"][i])

  return torch.tensor(input_ids),torch.tensor(attention_mask),torch.tensor(token_type_ids)

def train_dataloader(train_dataset):
  train_sampler = SequentialSampler(train_dataset)
  model_collate_fn = functools.partial(
    process_batch,
    tokenizer=tokenizer,
    max_len=args.max_seq_length
    )
  train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              collate_fn=model_collate_fn)
  return train_dataloader


set_seed(1)

#Loading dataset  
train_data = wikiDataset("data/wiki1m_for_simcse.txt", full=True)

train_dataloader = train_dataloader(train_data)

num_train_optimization_steps = int(len(train_data) / args.batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_data))
logger.info("  Num Epochs = %d", args.num_train_epochs)

model = BertForCL.from_pretrained(args.model_name_or_path,config=config)
model.to(device)
logger.info(model)

param_optimizer = list(model.named_parameters())
no_decay = ['bias','LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_optimization_steps)

for epoch in range(args.num_train_epochs):
  model.train()
  running_loss = 0.0
  for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
  #for input_ids,attention_mask,token_type_ids in train_dataloader:
    batch = tuple(t.to(device) for t in batch)
    input_ids,attention_mask,token_type_ids = batch
    #import ipdb; ipdb.set_trace();
    # zero the parameter gradients
    optimizer.zero_grad()
    outputs = model(input_ids,attention_mask,token_type_ids)
    loss = outputs["loss"]

    if args.gradient_accumulation_steps > 1:
      loss = loss / args.gradient_accumulation_steps
    loss.backward()
    running_loss += loss.item()
    if (step + 1) % args.gradient_accumulation_steps == 0:
      optimizer.step()
      scheduler.step()  # Update learning rate schedule
      model.zero_grad()
    optimizer.step()
    logger.info("Step Loss", loss.item())
  
  logger.info("Epoch Loss",running_loss)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
config.save_pretrained(args.output_dir)

