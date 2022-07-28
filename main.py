import os

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import onnx
import onnxruntime
from architectures import SimpleCNN
from datasets import LoadImages
from utils import plot


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, device: torch.device):

    if model == 0:
        model = torch.jit.load('results/best_model.pt')
    model.eval()
    loss = 0
    with torch.no_grad(): 
        for data in tqdm(dataloader, desc="scoring", position=0):
            targets, inputs = data

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss += torch.sqrt(loss_fn(outputs, targets))

    loss /= len(dataloader)
    model.train()
    return loss


def main(results_path, network_config: dict, learningrate, weight_decay, n_updates, device: torch.device = torch.device("cuda:0")):
    np.random.seed(0)
    torch.manual_seed(0)
    
    plotpath = os.path.join(results_path, "plots")
    dataset = LoadImages(folder='training')

    trainingset = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (3 / 5))))
    validationset = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (3 / 5)), int(len(dataset) * (4 / 5))))
    testset = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (4 / 5)), len(dataset)))

    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=64, shuffle=False, num_workers=4)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))
    
    net = SimpleCNN(**network_config)
    net.to(device)

    mse = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)
    
    print_stats_at = 100 
    plot_at = 10_000 
    validate_at = 5000
    update = 0 
    best_validation_loss = np.inf
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    saved_model_file = os.path.join(results_path, "best_model.pt")
    model_scripted = torch.jit.script(net)
    model_scripted.save(saved_model_file)

    while update < n_updates:
        for data in trainloader:

            targets, inputs = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            
            loss = torch.sqrt(mse(outputs, targets))
            loss.backward()
            optimizer.step()
            
            if (update + 1) % print_stats_at == 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)
            
            if (update + 1) % plot_at == 0:
                plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plotpath, update)
            
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(net, dataloader=valloader, loss_fn=mse, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param.cpu(), global_step=update)
                    writer.add_histogram(tag=f"validation/gradients_{i} ({name})", values=param.grad.cpu(), global_step=update)

                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    model_scripted = torch.jit.script(net)
                    model_scripted.save(saved_model_file)
            
            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()
            
            update += 1
            if update >= n_updates:
                break

    update_progress_bar.close()
    writer.close()
    
    print(f"Computing scores...")
    train_loss = evaluate_model(0, dataloader=trainloader, loss_fn=mse, device=device)
    val_loss = evaluate_model(0, dataloader=valloader, loss_fn=mse, device=device)
    test_loss = evaluate_model(0, dataloader=testloader, loss_fn=mse, device=device)
    
    print(f"Scores:")
    print(f"training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"test loss: {test_loss}")
    
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"test loss: {test_loss}", file=rf)


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file")
    args = parser.parse_args()
    
    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)
