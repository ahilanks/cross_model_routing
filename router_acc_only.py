import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
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

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

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

    "LENGTH": 7000,
    "NUM_UNFROZEN": 0,
    "PEAK_LR": 1e-4,
    "BATCH_SIZE": 32,
    "EPOCHS": 8,
    "WEIGHT_DECAY": 0.1,
    "NUM_WARMUP_STEPS": 50,
    "SAVE_PATH": "models",
    "DROPOUT": 0.4,
    "NUM_LAYERS": 8,
    "DATASET": "a5ilank/RouterBench2.0"
}


def load_data(dataset, model_name, length=None):
    ds = load_dataset(dataset, split="train")
    # if length is not None:
    #     ds = ds.select(range(length))

    df = ds.to_pandas()

    df_neg = df[df[model_name].apply(lambda d: d['is_correct'] is False)]
    df_pos = df[df[model_name].apply(lambda d: d['is_correct'] is True)]

    if length is not None:
        if len(df_neg) >= length:
            df_neg = df_neg.sample(length // 2)
            df_pos = df_pos.sample(length // 2)
        else:
            remaining = length - len(df_neg)
            df_pos = df_pos.sample(n=remaining, random_state=42, replace=(len(df_pos) < remaining))
        
    df = pd.concat([df_neg, df_pos], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)


    # Split into train and validation
    df_train = df.sample(frac=0.9, random_state=42)
    df_val = df.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_train, df_val = df_train[['question', model_name]], df_val[['question', model_name]]

    num_pos = sum(df_train[model_name].apply(lambda d: d['is_correct']))
    num_neg = len(df_train) - num_pos
    weight_value = num_neg / num_pos

    return df_train, df_val, weight_value
    #return df_train, df_val

# --- CHANGE: Removed lat and price extraction ---
def df_to_tensors(df, model_name):
    queries, t_acc = [], []
    for idx, row in df.iterrows():
        queries.append(row['question'])
        t_acc.append(row[model_name]['is_correct'])
    return queries, t_acc

class Router(nn.Module):
    def __init__(self, model_name, num_layers, dropout):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        self.hidden_size = self.base_model.config.hidden_size 

        layers = []
        input_dim = self.hidden_size
        
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, input_dim // 2))
            layers.append(nn.BatchNorm1d(input_dim // 2)) # Crucial for deep nets
            layers.append(nn.Dropout(dropout))
            layers.append(nn.GELU())
            input_dim = input_dim // 2
            
            if input_dim < 32:
                break

        # Final layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.acc_head = nn.Sequential(*layers)

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
    def __init__(self, pos_weight):
            super().__init__()
            # --- CHANGE: Removed MSELoss ---
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            #self.bce = nn.BCEWithLogitsLoss()

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

            wandb.log({
                "loss/val_acc": loss.item(),
            }, step=global_step)
            
            total_val_loss += loss.item()
            num_batches += 1

    val_auc = roc_auc_score(all_targets_acc, all_preds_acc)
    threshold = 0.5
    pred_labels = (np.array(all_preds_acc) >= threshold).astype(int)
    target_labels = np.array(all_targets_acc).astype(int)

    threshold_acc = (pred_labels == target_labels).mean()



    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Validation Threshold Accuracy: {threshold_acc:.4f}")
    wandb.log({"val/auc": val_auc, "val/acc@0.5": threshold_acc}, step=global_step)


    avg_loss = total_val_loss / num_batches

    print(f"Validation Loss (Epoch {epoch+1}): {avg_loss:.4f}")
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
            global_step += 1

        avg_loss = total_loss / (len(inputs['input_ids']) / batch_size)
        print(f"Epoch {epoch+1} done in {time.time()-start_time:.1f}s | Avg Loss: {avg_loss:.4f}")

        val_loss = validate(router, tokenizer, df_val, model_name, loss_fn, device, epoch, global_step)

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

def test_inference(router, tokenizer, device):
    """Test inference on 10 different queries."""
    test_queries = [
        "Explain why the sky is blue.",
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Write a Python function to calculate the factorial of a number.",
        "What are the main causes of climate change?",
        "Explain the difference between machine learning and deep learning.",
        "What is the time complexity of quicksort?",
        "Describe the process of DNA replication.",
        "How do neural networks learn?",
        "What is the difference between HTTP and HTTPS?",
    ]
    
    print("\n" + "="*60)
    print("Testing inference on 10 different queries:")
    print("="*60)
    
    for i, query in enumerate(test_queries, 1):
        result = infer(router, tokenizer, query, device)
        print(f"\nQuery {i}: {query}")
        print(f"Predicted Accuracy: {result['predicted_accuracy']:.4f}")
    
    print("\n" + "="*60)
    print("Inference testing complete.")
    print("="*60 + "\n")

def main():

    #setup()

    # local_rank = int(os.environ["LOCAL_RANK"])
    # device = torch.device(f"cuda:{local_rank}")


    model_name = "gemini-2.5-flash-lite_response"
    print("Loading data...")
    df_train, df_val, weight_value = load_data(CONFIG["DATASET"], model_name, CONFIG["LENGTH"])
    #df_train, df_val = load_data("a5ilank/RouterBench2.0", model_name, CONFIG["LENGTH"])

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

    router = Router(CONFIG["base"], num_layers=CONFIG["NUM_LAYERS"], dropout=CONFIG["DROPOUT"]).to(CONFIG["device"])
    
    pos_weight_tensor = torch.tensor([weight_value]).to(CONFIG["device"])

    for param in router.base_model.parameters():
        param.requires_grad = False

    # num_unfrozen = CONFIG["NUM_UNFROZEN"]
    # layers = router.base_model.layers
    # for layer in layers[-num_unfrozen:]:
    #     layer.to(torch.float32)
    #     for param in layer.parameters():
    #         param.requires_grad = True

    # trainable, frozen = 0, 0
    # for n, p in router.named_parameters():
    #     if p.requires_grad:
    #         trainable += p.numel()
    #     else:
    #         frozen += p.numel()
    # print(f"Trainable: {trainable:,}, Frozen: {frozen:,}")

    # router = torch.nn.parallel.DistributedDataParallel(
    #     router, 
    #     device_ids=[local_rank],
    #     output_device=local_rank
    # )

    opt = optim.AdamW(router.parameters(), lr=CONFIG["PEAK_LR"], weight_decay=CONFIG["WEIGHT_DECAY"])
    loss_fn = RouterLoss(pos_weight=pos_weight_tensor)
    #loss_fn = RouterLoss()
    num_training_steps = (len(inputs['input_ids']) // CONFIG["BATCH_SIZE"]) * CONFIG["EPOCHS"]
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=CONFIG["NUM_WARMUP_STEPS"], num_training_steps=num_training_steps)

    print("Training router...")
    train(router, tokenizer, inputs, targets, df_val, model_name, opt, loss_fn, scheduler, CONFIG["device"], epochs=CONFIG["EPOCHS"], batch_size=CONFIG["BATCH_SIZE"])

    
    save_dir = os.path.join(CONFIG["SAVE_PATH"], "gemini")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(router.state_dict(), os.path.join(save_dir, "router.pt"))
    tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
    print(f"Router weights and tokenizer saved to {save_dir}")
    

    # Test inference
    test_inference(router, tokenizer, CONFIG["device"])

    print("Finishing wandb...")
    time.sleep(10)
    wandb.finish()
    
if __name__ == "__main__":
    main()