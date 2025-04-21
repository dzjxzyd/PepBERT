from model import build_transformer
from dataset import PeptideDataset
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Split

import pandas as pd
from transformers import get_scheduler, get_constant_schedule_with_warmup

def run_validation(model, val_dataloader, tokenizer, device,loss_fn):
    model.eval()
    count = 0
    loss_val = 0
    with torch.no_grad():
        for batch in val_dataloader:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(encoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # # Compute the loss using a simple cross entropy
            # loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))

            # Masked Language Modeling Loss Calculation
            # Identify where the labels are masked tokens (assuming `[MASK]` token id is defined)
            mask_token_id = tokenizer.token_to_id("[MASK]")
            mask = encoder_input == mask_token_id  # mask is a boolean tensor indicating masked positions
            # Reshape the output and labels, but only keep the masked tokens
            proj_output_masked = proj_output.view(-1, tokenizer.get_vocab_size())[mask.view(-1)]
            label_masked = label.view(-1)[mask.view(-1)]
            # Compute the loss only for masked positions
            loss = loss_fn(proj_output_masked, label_masked)
    
            loss_val += loss.item() * proj_output_masked.shape[0] # only a limited seqeunces was mased somehow
            count += proj_output_masked.shape[0]

    ave_val_loss = loss_val / count

    return ave_val_loss


def get_or_build_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Split(pattern='', behavior='isolated') # Split by every character
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]","[MASK]"], min_frequency=2)
        tokenizer.train_from_iterator(ds, trainer=trainer) # it can read list 
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(f"{config['datasource']}", split='train')[config['sequence_column']]# i only need the sequence -output is a list
    
    # Build tokenizers
    tokenizer = get_or_build_tokenizer(config, ds_raw)
    # tokenizer_tgt = get_or_build_tokenizer(config, ds, config['lang_tgt'])

    # 0.1 % as the validation dataset for early stop
    val_ds_size = int(0.0001 * len(ds_raw))
    train_ds_size  = len(ds_raw) - val_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = PeptideDataset(train_ds_raw, tokenizer,  config['seq_len'])
    val_ds = PeptideDataset(val_ds_raw, tokenizer, config['seq_len'])
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1000, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer

def get_model(config, vocab_len):
    model = build_transformer(vocab_len, config["seq_len"],  d_model=config['d_model'])
    return model


def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    # in this optimizer, we only set one parameter group : model.parameters()
    # it is possible we separate the parameters inside the model to different groups and set different lr for them, repectively
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-8, betas=(0.9, 0.98), weight_decay=0.01)

    # custom learning rate schedule
    num_warmup_steps = 2000
    num_total_steps = config['num_epochs'] * len(train_dataloader)

    def update_lr(optimizer, step, num_warmup_steps, num_total_steps, base_lr, min_lr):
        if step < num_warmup_steps:
            # warmup to peak lr
            lr = base_lr * (step / num_warmup_steps)
        elif  num_warmup_steps < step < int(num_total_steps * 0.9):
            # linear decay to min_lr
            decay_steps = int(num_total_steps * 0.9) - num_warmup_steps
            decay_progress = (step - num_warmup_steps) / decay_steps
            lr = base_lr - decay_progress * (base_lr - min_lr)
        else: # keep the lr till end
            lr = base_lr / 10
        # update the lr for all the parameter groups, actually here we only have a single parameter group, model.parameters()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0).to(device)
    loss_train_col = []
    loss_val_col = []
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        loss_train = 0
        count = 0
        for batch in batch_iterator:
            
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)

            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(encoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # # Compute the loss using a simple cross entropy
            # loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            
            # Masked Language Modeling Loss Calculation
            # Identify where the labels are masked tokens (assuming `[MASK]` token id is defined)
            mask_token_id = tokenizer.token_to_id("[MASK]")
            mask = encoder_input == mask_token_id  # mask is a boolean tensor indicating masked positions
            # Reshape the output and labels, but only keep the masked tokens
            proj_output_masked = proj_output.view(-1, tokenizer.get_vocab_size())[mask.view(-1)]
            label_masked = label.view(-1)[mask.view(-1)]
            # Compute the loss only for masked positions
            loss = loss_fn(proj_output_masked, label_masked)
    
            loss_train += loss.item() * proj_output_masked.shape[0] # only a limited seqeunces was mased somehow
            count += proj_output_masked.shape[0]
            
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            global_step += 1
            

            # Backpropagate the loss
            loss.backward()
            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)        
            update_lr(optimizer, global_step, num_warmup_steps, num_total_steps, config['lr'],config['lr']/10  )  
        print(optimizer.param_groups[0]['lr'])
        # Run validation at the end of every epoch
        ave_val_loss = run_validation(model, val_dataloader, tokenizer, device,loss_fn)
        print(f'train los in {epoch} is {loss_train/count}')
        print(f'val loss in {epoch} is {ave_val_loss}')
        loss_train_col.append(loss_train/count)
        loss_val_col.append(ave_val_loss)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
    
    # Creating a DataFrame from the lists
    df = pd.DataFrame({
        "loss_train": loss_train_col,
        "loss_val": loss_val_col
    })
    # Saving the DataFrame to a CSV file
    data_source = config['datasource'].split('/')[-1]
    filename = data_source+'loss.csv'
    df.to_csv(filename, index=False)



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
