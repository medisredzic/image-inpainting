# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.utils.data
from utils import plot
from architectures import SimpleCNN
import dill as pkl
import tqdm


def main(load_model: str, 
         numpy_dataset: str, 
         output_folder: str, 
         network_config: dict,
         device: torch.device = torch.device("cuda:0")):

    net = SimpleCNN(**network_config)
    net.to(device)
    net = torch.jit.load(load_model, map_location=torch.device('cuda'))

    with open(numpy_dataset, 'rb') as f:
        numpy_dataset = pkl.load(f)  # input_arrays, known_arrays, offsets, spacings, sample_ids
        n_samples = len(numpy_dataset['input_arrays'])

    preds = []
    with torch.no_grad():
        for sample_i in tqdm.tqdm(range(n_samples), desc="Creating predictions"):

            image = np.asarray(numpy_dataset['input_arrays'][sample_i], dtype=np.float32)

            inputs = image[None]
            inputs = torch.from_numpy(inputs).to(device)
            outputs = net(inputs)

            prediction = outputs[0].detach().cpu().numpy()
            prediction = np.clip(prediction, a_min=0, a_max=255)
            prediction = np.asarray(prediction, dtype=np.uint8)

            target_arr = prediction[numpy_dataset['known_arrays'][sample_i] == 0].copy()

            preds.append(target_arr)
    
    with open(os.path.join(output_folder, "predictions.pkl"), "wb") as ufh:
        pkl.dump(predictions, file=ufh)


if __name__ == '__main__':
    main('results/best_model.pt', 'test/inputs.pkl', 'output', 'working_config.json')
