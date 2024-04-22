import torch
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_step(model, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    model_outputs = model(**inputs)
    model_outputs.loss.backward()
    optimizer.step()
    return model_outputs.loss


def train_epoch(predictor, dataloader: DataLoader, optimizer: Optimizer, device: str):
    predictor.model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    for item in pbar:
        page, page_size = item
        model_inputs = predictor.preprocess_pdf_data(page, page_size, False)
        batched_inputs = predictor.model_input_collator(model_inputs, 1)
        for batch in batched_inputs:
            train_loss = train_step(predictor.model, batch, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")


def train(predictor, dataloader, num_epoch, device):
    optimizer = AdamW(predictor.model.parameters(), lr=0.001)
    for _ in range(num_epoch):
        train_epoch(predictor, dataloader, optimizer, device)