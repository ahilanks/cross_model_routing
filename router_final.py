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
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

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

    "PEAK_LR": 1e-6,
    "BATCH_SIZE": 16,
    "EPOCHS": 8,
    "WEIGHT_DECAY": 0.04,
    "NUM_WARMUP_STEPS": 50,
    "SAVE_PATH": "models/router_qwen2.5"
}


def load_data(file_path, length=None):
    records = []
    with open(file_path, "r") as f:
        for line in f:
            if length and len(records) >= length:
                break
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print("Bad line:", e)
    df = pd.DataFrame(records)

    # merge by model keys
    grouped = df.groupby("index")
    merged_rows = []
    for idx, group in grouped:
        merged_row = group.iloc[0].copy()
        for m in CONFIG["model_keys"]:
            vals = group[m].dropna().tolist()
            if vals:
                merged_row[m] = vals[-1]
        merged_rows.append(merged_row)
    df = pd.DataFrame(merged_rows).reset_index(drop=True)

    # clean & get is_correct, and total_price
    for idx, row in df.iterrows():
        for col in CONFIG["model_keys"].keys():
            if pd.notna(row.get(col)) and isinstance(row[col], dict):
                entry = row[col].copy()

                # parse final answer
                answer = entry.get('answer', '')
                if answer and 'Final Answer:' in answer:
                    final_answer_str = answer.split("Final Answer:")[1].strip()
                    try:
                        final_answer = int(final_answer_str)
                    except ValueError:
                        final_answer = 0
                    entry['final_answer'] = final_answer

                    correct_answer = row.get('correct_answer', '')
                    entry['is_correct'] = (final_answer == correct_answer)

                # compute total price
                input_tokens = entry.get('input_tokens', 0)
                output_tokens = entry.get('output_tokens', 0)
                input_rate, output_rate = CONFIG["model_keys"][col]
                entry['total_price'] = (input_tokens * (input_rate / 1000000) + output_tokens * (output_rate / 1000000)) * 1000 # kind of normalizing

                # default missing numeric fields to 0
                entry['total_latency'] = entry.get('total_latency', 0) or 0
                entry['is_correct'] = entry.get('is_correct', False)

                df.at[idx, col] = entry

    # Drop rows with missing or invalid question
    df = df[df['question'].notna() & (df['question'] != '')].reset_index(drop=True)

    # Ensure no nulls in key model fields (example: gemini)
    key_model = 'gemini-2.5-flash-lite_response'
    df = df[
        df[key_model].apply(lambda x: isinstance(x, dict) and all(k in x and x[k] is not None for k in ['is_correct', 'total_latency', 'total_price']))
    ].reset_index(drop=True)

    # Split into train and validation
    df_train = df.sample(frac=0.9, random_state=42)
    df_val = df.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)

    return df_train, df_val

def df_to_tensors(df):
    queries, t_acc, t_lat, t_price = [], [], [], []
    for idx, row in df.iterrows():
        queries.append(row['question'])
        t_acc.append(row['gemini-2.5-flash-lite_response']['is_correct'])
        t_lat.append(np.log1p(row['gemini-2.5-flash-lite_response']['total_latency'])) #log tranform for latency and price
        t_price.append(np.log1p(row['gemini-2.5-flash-lite_response']['total_price']))
    return queries, t_acc, t_lat, t_price

class Router(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        self.hidden_size = self.base_model.config.hidden_size # same hidden size as base

        # FIX: Remove dtype=torch.float16 from head layers for GradScaler stability
        self.acc_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size // 2), nn.GELU(), nn.Linear(self.hidden_size // 2, 1))
        self.lat_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size // 2), nn.GELU(), nn.Linear(self.hidden_size // 2, 1), nn.Softplus())
        self.price_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size // 2), nn.GELU(), nn.Linear(self.hidden_size // 2, 1), nn.Softplus())


    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        shared_embedding = outputs.last_hidden_state[:, -1, :]

        acc = self.acc_head(shared_embedding)
        lat = self.lat_head(shared_embedding)
        price = self.price_head(shared_embedding)

        return {'acc': acc, 'lat': lat, 'price': price}

class RouterLoss(nn.Module):
    def __init__(self, weight_acc=1.0, weight_lat=1.0, weight_price=1.0):
            super().__init__()
            self.weights = (weight_acc, weight_lat, weight_price)
            # BCEWithLogitsLoss is the correct, stable loss for logits
            self.bce, self.mse = nn.BCEWithLogitsLoss(), nn.MSELoss() 

    def forward(self, preds, targets):
        total_loss = 0
        # targets are now expected to be float32
        loss_acc = self.bce(
            preds['acc'], 
            targets['acc']
        )
        loss_lat = self.mse(
            preds['lat'],
            targets['lat']
        )
        loss_price = self.mse(
            preds['price'],
            targets['price']
        )
        total_loss = self.weights[0] * loss_acc + self.weights[1] * loss_lat + self.weights[2] * loss_price
        return total_loss


def validate(router, tokenizer, df_val, loss_fn, device, writer=None, epoch=0):
    router.eval()
    queries, t_acc, t_lat, t_price = df_to_tensors(df_val)
    inputs = tokenizer(queries, padding=True, truncation=True, return_tensors='pt')
    
    # FIX: Targets should be float32 for stable loss calculation
    targets = {
        'acc': torch.tensor(t_acc, dtype=torch.float32).unsqueeze(1),
        'lat': torch.tensor(t_lat, dtype=torch.float32).unsqueeze(1),
        'price': torch.tensor(t_price, dtype=torch.float32).unsqueeze(1)
    }

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
                loss = loss_fn(preds, b_target)
            
            total_val_loss += loss.item()
            num_batches += 1

    avg_loss = total_val_loss / num_batches

    print(f"Validation Loss (Epoch {epoch+1}): {avg_loss:.4f}")
    if writer:
        writer.add_scalar("Loss/val", avg_loss, epoch + 1)
    return avg_loss


def train(router, tokenizer, inputs, targets, df_val, opt, loss_fn, scheduler, device, epochs, batch_size):
    writer = SummaryWriter(log_dir="runs/router_training")
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
                loss = loss_fn(preds, b_target)
            
            # 1. Scale loss and backward pass
            scaler.scale(loss).backward()
            
            # 2. Unscale gradients (required before clipping)
            # This is safe now because head parameters are float32
            scaler.unscale_(opt)

            # 3. Clip gradients on the unscaled (float32) gradients
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)

            # 4. Step optimizer and update scaler
            scaler.step(opt)

            scheduler.step()

            scaler.update()

            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], global_step)

            total_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1

        avg_loss = total_loss / (len(inputs['input_ids']) / batch_size)
        print(f"Epoch {epoch+1} done in {time.time()-start_time:.1f}s | Avg Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch+1)

        val_loss = validate(router, tokenizer, df_val, loss_fn, device, writer, epoch)
        print(f"Validation Loss: {val_loss:.4f}")

    writer.close()
    print("TensorBoard logs written to 'runs/router_training'.")


def infer(router, tokenizer, prompt, device):
    router.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        # Use autocast for inference to match training precision, though not strictly required here
        with torch.amp.autocast('cuda'):
            preds = router(inputs["input_ids"], inputs["attention_mask"])
            
    return {
        "predicted_accuracy": torch.sigmoid(preds['acc']).item(),
        "predicted_latency": np.expm1(preds['lat'].item()), # revert log transform for latency and price
        "predicted_price": np.expm1(preds['price'].item()), 
    }

def main():
    print("Loading data...")
    df_train, df_val = load_data("data/all_providers_results_parallelized.jsonl", length=5000)
    df_train, df_val = df_train[['question', 'gemini-2.5-flash-lite_response']], df_val[['question', 'gemini-2.5-flash-lite_response']]

    print("Training data size:", len(df_train), "Validation data size:", len(df_val))
    
    queries, t_acc, t_lat, t_price = df_to_tensors(df_train)

    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base"])
    inputs = tokenizer(queries, padding=True, truncation=True, return_tensors='pt').to(CONFIG["device"])

    # FIX: Targets should be float32 for stable loss calculation
    targets = {
        'acc': torch.tensor(t_acc, dtype=torch.float32).unsqueeze(1).to(CONFIG["device"]),
        'lat': torch.tensor(t_lat, dtype=torch.float32).unsqueeze(1).to(CONFIG["device"]),
        'price': torch.tensor(t_price, dtype=torch.float32).unsqueeze(1).to(CONFIG["device"])
    }

    router = Router(CONFIG["base"]).to(CONFIG["device"])

    for param in router.base_model.parameters():
        param.requires_grad = False

    num_unfrozen = 4
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
    loss_fn = RouterLoss()
    num_training_steps = (len(inputs['input_ids']) // CONFIG["BATCH_SIZE"]) * CONFIG["EPOCHS"]
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=CONFIG["NUM_WARMUP_STEPS"], num_training_steps=num_training_steps)

    print("Training router...")
    train(router, tokenizer, inputs, targets, df_val, opt, loss_fn, scheduler, CONFIG["device"], epochs=CONFIG["EPOCHS"], batch_size=CONFIG["BATCH_SIZE"])

    
    os.makedirs(CONFIG["SAVE_PATH"], exist_ok=True)

    torch.save(router.state_dict(), os.path.join(CONFIG["SAVE_PATH"], "router.pt"))
    tokenizer.save_pretrained(CONFIG["SAVE_PATH"])
    print(f"Router weights and tokenizer saved to {CONFIG["SAVE_PATH"]}")

    # Test inference
    test_query = "Explain why the sky is blue."
    print("Example inference:", infer(router, tokenizer, test_query, CONFIG["device"]))
    
if __name__ == "__main__":
    main()