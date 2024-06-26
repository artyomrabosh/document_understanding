import torch
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from vila import HierarchicalPDFPredictor
from tqdm import tqdm

from typing import List

import wandb

from sklearn.metrics import classification_report

model_config = {
    'trainable_params': ['classifier', 'layout_encoder', 'text_encoder']
}

def train_step(model: torch.nn.Module, 
               inputs: torch.Tensor, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device) -> torch.Tensor:
    """
    Perform a single training step.

    Args:
        model (torch.nn.Module): The model to train.
        inputs (torch.Tensor): The input tensor.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (str): The device to use for computation ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The loss tensor.
    """
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model_outputs = model(**inputs)
    model_outputs.loss.backward()
    return model_outputs.loss

def compute_metrics(preds, labels):
    report = classification_report(labels, preds, zero_division=0.0, output_dict=True)
    wandb.log(report)

def set_training_parameters(predictor: HierarchicalPDFPredictor, 
                            trainable_params: List[str]):
    if "classifier" in trainable_params:
        predictor.model.classifier.train()
    if "text_encoder" in trainable_params:
        predictor.model.hierarchical_model.textline_encoder.train()
    if "layout_encoder" in trainable_params:
        predictor.model.hierarchical_model.textline_model.train()
    if "layout_embeddings" in trainable_params:
        predictor.model.hierarchical_model.textline_model.embeddings.train()



def train_epoch(predictor: HierarchicalPDFPredictor, 
                dataloader: DataLoader, 
                optimizer: Optimizer, 
                device: torch.device = torch.device('cpu'), 
                accumulation_grad_steps: int = 32) -> None:
    """
    Train the model for a single epoch.

    Args:
        predictor (PDFPredictor): The predictor object containing the model and preprocessing functions.
        dataloader (DataLoader): The DataLoader to use for training.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (str): The device to use for computation ('cpu' or 'cuda').
    """
    predictor.model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    optimizer.zero_grad()
    step = 0
    for _, item in enumerate(pbar):
        page, page_size = item
        model_inputs = predictor.preprocess_pdf_data(page, page_size, False)
        del page
        del page_size
        batched_inputs = predictor.model_input_collator(model_inputs, 1)
        del model_inputs
        for batch in batched_inputs:
            train_loss = train_step(predictor.model, batch, optimizer, device)
            step += 1
            if step % accumulation_grad_steps == 0:
                wandb.log({"train loss": train_loss})
                optimizer.step()
                optimizer.zero_grad()
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")

def validate(predictor: HierarchicalPDFPredictor, 
             dataloader: DataLoader, 
             device: torch.device = torch.device('cpu')):
    """
    Validate the model on the validation set.

    Args:
        predictor (PDFPredictor): The predictor object containing the model and preprocessing functions.
        dataloader (DataLoader): The DataLoader to use for validation.
    """
    device = torch.device(device)
    predictor.model.eval()
    predictor.model.to(device)
    pbar = tqdm(dataloader)
    preds, labels = [], []

    for _, item in enumerate(pbar):
        page, page_size = item
        model_inputs = predictor.preprocess_pdf_data(page, page_size, False)
        batched_inputs = predictor.model_input_collator(model_inputs, 1)
        model_predictions = []
        batch_losses = []
        for batch in batched_inputs:
            batch = {k: v.to(device) for k, v in batch.items()}
            model_outputs = predictor.model(**batch)
            batch_losses.append(model_outputs.loss)
            model_predictions.append(predictor.get_category_prediction(model_outputs))
        model_predictions = predictor.postprocess_model_outputs(page, model_inputs, model_predictions, 'list')
        model_predictions = [predictor.id2label[pred] for pred in model_predictions]
        item_labels = [predictor.id2label[label] for label in page['labels']]

        preds.extend(model_predictions)
        labels.extend(item_labels)
    compute_metrics(preds, labels)


def train(predictor: HierarchicalPDFPredictor, dataloader: DataLoader, num_epoch: int, device: str = 'cpu'):
    """
    Train the model for a specified number of epochs.

    Args:
        predictor (PDFPredictor): The predictor object containing the model and preprocessing functions.
        dataloader (DataLoader): The DataLoader to use for training.
        num_epoch (int): The number of epochs to train for.
        device (str): The device to use for computation ('cpu' or 'cuda').
    """
    optimizer = AdamW(predictor.model.parameters(), lr=0.0001)
    device = torch.device(device)
    predictor.model.to(device)

    for _ in range(num_epoch):
        train_epoch(predictor, dataloader, optimizer, device)