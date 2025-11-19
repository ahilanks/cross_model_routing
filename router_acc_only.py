import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import json
import re
import pandas as pd
import os
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import wandb
from datasets import load_dataset
from datetime import datetime

CONFIG = {
    "base": "Qwen/Qwen2.5-1.5B-Instruct", # base model
    "device": torch.device("cuda"),

    # model names and prices
    "model_keys": {
        'gpt-5-nano-2025-08-07_response': (0.05, 0.40),
        'grok-4-fast-reasoning_response': (0.20, 0.50),
        'openai_gpt-oss-120b_response': (0.05, 0.24),
        'openai_gpt-oss-20b_response': (0.03, 0.14),
        'nvidia_NVIDIA-Nemotron-Nano-9B-v2_response': (0.04, 0.16),
        'meta-llama_Llama-3.2-11B-Vision-Instruct_response': (0.049, 0.049),
        'moonshotai_Kimi-K2-Instruct-0905_response': (0.50, 2.00),
        'gemini-2.5-flash-lite_response': (0.10, 0.40),
    },

    "NUM_UNFROZEN": 3,
    "PEAK_LR": 5e-6,
    "BATCH_SIZE": 32,
    "EPOCHS": 5,
    "WEIGHT_DECAY": 0.1,
    "NUM_WARMUP_STEPS": 50,
    "SAVE_PATH": "models/router_qwen2.5"
}


def load_data(dataset, model_name, length=None):
    ds = load_dataset(dataset, split="train")
    if length is not None:
        ds = ds.select(range(length))

    df = ds.to_pandas()

    # Split into train and validation
    df_train = df.sample(frac=0.9, random_state=42)
    df_val = df.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_train, df_val = df_train[['question', model_name]], df_val[['question', model_name]]

    num_pos = sum(df_train[model_name].apply(lambda d: d['is_correct']))
    num_neg = len(df_train) - num_pos
    weight_value = num_neg / num_pos

    return df_train, df_val, weight_value

# --- CHANGE: Removed lat and price extraction ---
def df_to_tensors(df, model_name):
    queries, t_acc = [], []
    for idx, row in df.iterrows():
        queries.append(row['question'])
        t_acc.append(row[model_name]['is_correct'])
    return queries, t_acc

class Router(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        self.hidden_size = self.base_model.config.hidden_size 

        # --- CHANGE: Removed lat_head and price_head ---
        self.acc_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size // 2), nn.GELU(), nn.Linear(self.hidden_size // 2, 1))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        shared_embedding = outputs.last_hidden_state[:, -1, :]

        acc = self.acc_head(shared_embedding)
        
        # --- CHANGE: Removed lat and price calculation ---
        return {'acc': acc}

class RouterLoss(nn.Module):
    # --- CHANGE: Removed weight_lat and weight_price arguments ---
    def __init__(self, pos_weight=None):
            super().__init__()
            # --- CHANGE: Removed MSELoss ---
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, preds, targets):
        # --- CHANGE: Removed lat and price loss calculation and weighting ---
        loss_acc = self.bce(
            preds['acc'], 
            targets['acc']
        )
        return loss_acc


def validate(router, tokenizer, df_val, model_name, loss_fn, device, epoch=0, global_step=0):
    router.eval()
    # --- CHANGE: Unpack only queries and acc ---
    queries, t_acc = df_to_tensors(df_val, model_name)
    inputs = tokenizer(queries, padding=True, truncation=True, return_tensors='pt')

    # --- CHANGE: Removed lat and price targets ---
    targets = {
        'acc': torch.tensor(t_acc, dtype=torch.float32).unsqueeze(1),
    }

    all_preds_acc = []
    all_targets_acc = []

    batch_size = CONFIG["BATCH_SIZE"]
    total_val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(inputs['input_ids']), batch_size), desc="Validation"):
            idx = slice(i, i + batch_size)
            
            # Move batch data to device
            b_input = {k: v[idx].to(device) for k, v in inputs.items()}
            b_target = {k: v[idx].to(device) for k, v in targets.items()}

            # Autocast is used for the base model's float16 weights
            with torch.amp.autocast('cuda'):
                preds = router(b_input['input_ids'], b_input['attention_mask'])
                # --- CHANGE: Loss function now returns single value ---
                loss = loss_fn(preds, b_target)
                all_preds_acc.extend(torch.sigmoid(preds['acc']).detach().float().cpu().numpy())
                all_targets_acc.extend(b_target['acc'].detach().float().cpu().numpy())

            # --- CHANGE: Removed logging for lat and price ---
            wandb.log({
                "loss/val_total": loss.item(),
                "loss/val_acc": loss.item(), # total is same as acc now
            }, step=global_step)
            
            total_val_loss += loss.item()
            num_batches += 1

    try:
        val_auc = roc_auc_score(all_targets_acc, all_preds_acc)
    except ValueError:
        val_auc = 0.5 # Handle single-class edge cases

    print(f"Validation AUC: {val_auc:.4f}")
    wandb.log({"val/auc": val_auc}, step=global_step)


    avg_loss = total_val_loss / num_batches

    print(f"Validation Loss (Epoch {epoch+1}): {avg_loss:.4f}")
    wandb.log({"loss/val": avg_loss}, step=epoch)
    return avg_loss


def train(router, tokenizer, inputs, targets, df_val, model_name, opt, loss_fn, scheduler, device, epochs, batch_size):
    router.train()
    global_step = 0

    # GradScaler is essential for stable mixed-precision training
    scaler = torch.amp.GradScaler()

    for epoch in range(epochs):
        perm = torch.randperm(len(inputs['input_ids']))
        total_loss = 0.0
        start_time = time.time()

        for i in tqdm(range(0, len(inputs['input_ids']), batch_size), desc=f"Epoch {epoch+1}"):
            idx = perm[i:i+batch_size]
            if len(idx) < batch_size and i > 0:
                 continue

            b_input = {k: v[idx] for k, v in inputs.items()}
            b_target = {k: v[idx] for k, v in targets.items()}

            opt.zero_grad()
            
            # Autocast runs the forward pass in float16 where safe
            with torch.amp.autocast('cuda'):
                preds = router(b_input['input_ids'], b_input['attention_mask'])
                # --- CHANGE: Loss function return unpacked ---
                loss = loss_fn(preds, b_target)

            # --- CHANGE: Wandb logging update ---
            wandb.log({
                "loss/train_total": loss.item(),
                "loss/train_acc": loss.item(),
                "lr": scheduler.get_last_lr()[0]
            }, step=global_step)
            
            # 1. Scale loss and backward pass
            scaler.scale(loss).backward()
            
            # 2. Unscale gradients (required before clipping)
            scaler.unscale_(opt)

            # 3. Clip gradients on the unscaled (float32) gradients
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)

            # 4. Step optimizer and update scaler
            scaler.step(opt)

            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            wandb.log({"loss/train": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=global_step)
            global_step += 1

        avg_loss = total_loss / (len(inputs['input_ids']) / batch_size)
        print(f"Epoch {epoch+1} done in {time.time()-start_time:.1f}s | Avg Loss: {avg_loss:.4f}")
        wandb.log({"loss/epoch_avg": avg_loss}, step=global_step)

        val_loss = validate(router, tokenizer, df_val, model_name, loss_fn, device, epoch, global_step)
        print(f"Validation Loss: {val_loss:.4f}")

def infer(router, tokenizer, prompt, device):
    router.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            preds = router(inputs["input_ids"], inputs["attention_mask"])
            
    return {
        "predicted_accuracy": torch.sigmoid(preds['acc']).item(),
        # --- CHANGE: Removed latency and price return ---
    }

def main():
    model_name = "gemini-2.5-flash-lite_response"
    print("Loading data...")
    df_train, df_val, weight_value = load_data("a5ilank/RouterBench2.0", model_name)
    

    print("Training data size:", len(df_train), "Validation data size:", len(df_val))
    print(f"Using pos_weight: {weight_value:.4f} to handle class imbalance.")

    wandb.init(
        project="router_qwen2.5",
        entity="ahilan-uc-berkeley-electrical-engineering-computer-sciences",
        config=CONFIG,
        name = f"accuracy_only_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    
    # --- CHANGE: Unpacking only queries and acc ---
    queries, t_acc = df_to_tensors(df_train, model_name)

    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base"])
    inputs = tokenizer(queries, padding=True, truncation=True, return_tensors='pt').to(CONFIG["device"])

    # --- CHANGE: Targets dict only contains acc ---
    targets = {
        'acc': torch.tensor(t_acc, dtype=torch.float32).unsqueeze(1).to(CONFIG["device"]),
    }

    router = Router(CONFIG["base"]).to(CONFIG["device"])
    pos_weight_tensor = torch.tensor([weight_value]).to(CONFIG["device"])

    for param in router.base_model.parameters():
        param.requires_grad = False

    num_unfrozen = CONFIG["NUM_UNFROZEN"]
    layers = router.base_model.layers
    for layer in layers[-num_unfrozen:]:
        layer.to(torch.float32)
        for param in layer.parameters():
            param.requires_grad = True

    trainable, frozen = 0, 0
    for n, p in router.named_parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    print(f"Trainable: {trainable:,}, Frozen: {frozen:,}")


    opt = optim.AdamW(router.parameters(), lr=CONFIG["PEAK_LR"], weight_decay=CONFIG["WEIGHT_DECAY"])
    loss_fn = RouterLoss(pos_weight=pos_weight_tensor)
    num_training_steps = (len(inputs['input_ids']) // CONFIG["BATCH_SIZE"]) * CONFIG["EPOCHS"]
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=CONFIG["NUM_WARMUP_STEPS"], num_training_steps=num_training_steps)

    print("Training router...")
    train(router, tokenizer, inputs, targets, df_val, model_name, opt, loss_fn, scheduler, CONFIG["device"], epochs=CONFIG["EPOCHS"], batch_size=CONFIG["BATCH_SIZE"])

    
    os.makedirs(CONFIG["SAVE_PATH"], exist_ok=True)

    torch.save(router.state_dict(), os.path.join(CONFIG["SAVE_PATH"], "router.pt"))
    tokenizer.save_pretrained(CONFIG["SAVE_PATH"])
    print(f"Router weights and tokenizer saved to {CONFIG['SAVE_PATH']}")
    wandb.finish()

    # Test inference
    test_query = "Explain why the sky is blue."
    print("Example inference:", infer(router, tokenizer, test_query, CONFIG["device"]))
    
if __name__ == "__main__":
    main()