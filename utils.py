import os
import numpy as np
from matplotlib import pyplot as plt


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    
    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            #ax.imshow(data[i, 0], cmap="gray", interpolation="none")
            ax.imshow(np.transpose(data[i].astype('uint8'), (1, 2, 0))[:, :, 0:3], interpolation='none')
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)
    
    plt.close(fig)


"""def _plot_predicted(image_arrays, predictions, mask):
    fig, ax = plt.subplots(3, 1)

    for i, (image, prediction, mask) in enumerate(zip(image_arrays, predictions, masks)):
        ax[0].clear()
        ax[0].set_title(f'Image_{i}')
        ax[0].imshow(image, cmap=plt.cm.gray, interpolation='none')
        ax[0].set_axis_off()

        ax[1].clear()
        ax[1].set_title(f'Prediction_{i}')
        ax[1].imshow(prediction, cmap=plt.cm.gray, interpolation='none')
        ax[1].set_axis_off()

        fig.tight_layout()"""