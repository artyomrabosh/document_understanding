import torch
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from vila import HierarchicalPDFPredictor
from tqdm import tqdm

def train_step(model: torch.nn.Module, inputs: torch.Tensor, optimizer: torch.optim.Optimizer, device: str) -> torch.Tensor:
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
    optimizer.zero_grad()
    model_outputs = model(**inputs)
    model_outputs.loss.backward()
    optimizer.step()
    return model_outputs.loss

def compute_metrics(inputs, outputs):
    pass

def train_epoch(predictor: HierarchicalPDFPredictor, dataloader: DataLoader, optimizer: Optimizer, device: str):
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
    for item in pbar:
        page, page_size = item
        model_inputs = predictor.preprocess_pdf_data(page, page_size, False)
        batched_inputs = predictor.model_input_collator(model_inputs, 1)
        for batch in batched_inputs:
            train_loss = train_step(predictor.model, batch, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")

def validate(predictor: HierarchicalPDFPredictor, dataloader: DataLoader):
    """
    Validate the model on the validation set.

    Args:
        predictor (PDFPredictor): The predictor object containing the model and preprocessing functions.
        dataloader (DataLoader): The DataLoader to use for validation.
    """
    predictor.model.eval()
    pbar = tqdm(dataloader)
    losses = []
    metrics = []
    for item in pbar:
        page, page_size = item
        model_inputs = predictor.preprocess_pdf_data(page, page_size, False)
        batched_inputs = predictor.model_input_collator(model_inputs, 1)
        model_predictions = []
        batch_losses = []
        for batch in batched_inputs:
            model_outputs = train_step(predictor.model, batch, optimizer, device)
            batch_losses.append(model_outputs.loss)
            model_predictions.append(pdf_predictor.get_category_prediction(model_outputs))
        model_predictions = pdf_predictor.postprocess_model_outputs(ba, model_inputs, model_predictions, 'list')
        metrics.append(compute_metrics(page, model_predictions))


def train(predictor: HierarchicalPDFPredictor, dataloader: DataLoader, num_epoch: int, device: str):
    """
    Train the model for a specified number of epochs.

    Args:
        predictor (PDFPredictor): The predictor object containing the model and preprocessing functions.
        dataloader (DataLoader): The DataLoader to use for training.
        num_epoch (int): The number of epochs to train for.
        device (str): The device to use for computation ('cpu' or 'cuda').
    """
    optimizer = AdamW(predictor.model.parameters(), lr=0.001)
    for _ in range(num_epoch):
        train_epoch(predictor, dataloader, optimizer, device)
        # validate(predictor, dataloader)