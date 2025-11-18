
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


if __name__ == "__main__":
    router = Router(CONFIG["base"]).to(CONFIG["device"])
    router.load_state_dict(torch.load(CONFIG["SAVE_PATH"] + "/router.pt"))
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base"])
    prompt = "What's a synonym for 'happy'?"
    print(infer(router, tokenizer, prompt, CONFIG["device"]))