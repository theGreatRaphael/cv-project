from dataclasses import dataclass
from pathlib import Path
from typing import Union
import torch
from tqdm import tqdm
import yaml

from torch.nn import MSELoss
import torch.optim as optim
from torch.optim import Adam, Optimizer

from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FocusStackingDataset

from network import CVNetwork

TRAIN_DATA_PATH = "data/batch_20230912_part1"


@dataclass
class Hyperparameters:
    num_input_images: int
    batch_size: int
    epochs: int
    learning_rate: float
    # TODO: maybe we want to configure more?


def load_hyperparameters(
    params_file: Union[str, Path] = "hyperparameters.yml"
) -> Hyperparameters:
    with open(params_file) as pf:
        params_dict = yaml.safe_load(pf)

    flat_params = {}
    for category in params_dict:
        flat_params.update(params_dict[category])

    return Hyperparameters(**flat_params)


def init_model(params: Hyperparameters):
    return CVNetwork(params.num_input_images)


def create_dataloader(
    params: Hyperparameters, path: Path = TRAIN_DATA_PATH
) -> DataLoader:
    
    transform = transforms.Compose([
        # TODO: maybe increased contrast would rock?!
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])

    dataset = FocusStackingDataset(root_dir=path, transform=transform)

    return DataLoader(dataset=dataset, batch_size=params.batch_size)


def train_epoch(
    model: CVNetwork,
    train_loader: DataLoader,
    loss_fun: MSELoss,
    optimizer: Optimizer,
    device: str
):
    model.train()

    losses = []
    
    for X, Y in tqdm(train_loader):
        X = [x.to(device) for x in X]
        Y = Y.to(device)
        
        optimizer.zero_grad()

        preds = model(X)

        loss = loss_fun(preds, Y)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return sum(losses) / len(losses)


def eval():
    # TODO: do we need it?? or is "overfitting" best for a good grade?
    pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    best_loss = None

    params = load_hyperparameters()

    model = init_model(params)
    model = model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fun = MSELoss()

    train_loader = create_dataloader(params=params, path=TRAIN_DATA_PATH)
    val_loader = None  # TODO

    for epoch in tqdm(range(params.epochs)):
        train_loss = train_epoch(
                model=model,
                train_loader=train_loader,
                loss_fun=loss_fun,
                optimizer=optimizer,
                device=device
            )
        
        if best_loss is None:
            # no need to safe model in first epoch either way
            best_loss = train_loss
        
        elif best_loss > train_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), f"{train_loss}_model.pt")
        
        tqdm.write(f"Epochs loss: {train_loss}")

if __name__ == "__main__":
    main()