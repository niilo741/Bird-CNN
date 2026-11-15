# Bird-CNN

A small convolutional neural network that recognises bird species from their sound. Recordings come from [Xeno-canto](https://xeno-canto.org), are cut into 5-second log-mel spectrogram chunks, and are classified by a CNN written in PyTorch.

The model currently knows 10 species (listed in [data/classes.json](data/classes.json)). It always answers with one of them, also for silence or a bird it has never heard.

## Installation

Tested with Python 3.14 and PyTorch 2.15

```bash
git clone https://github.com/niilo741/Bird-CNN.git
cd Bird-CNN
conda create -n bird-cnn python=3.14
conda activate bird-cnn
pip install torch librosa numpy pandas scikit-learn requests python-dotenv matplotlib ipykernel
```

Downloading recordings needs a Xeno-canto API key (free, from your Xeno-canto account page). Put it in a file named `.env` in the project root:

```
API_KEY=your-key-here
```

## Usage

Run every command from the project root, as a module (`python -m src.<name>`).

| Step | Command | What it does |
|---|---|---|
| 1. Download | `python -m src.downloader` | Fetches up to 150 recordings per species into `data/raw/` and lists them in `data/metadata.csv` |
| 2. Preprocess | `python -m src.preprocess` | Cuts recordings into 5 s chunks and saves log-mel spectrograms to `data/processed/` |
| 3. Train | `python -m src.train` | Trains for 40 epochs, saves the best epoch to `checkpoints/best.pt`, prints a test report |
| 4. Predict | `python -m src.predict path/to/audio.mp3` | Prints the three most likely species for a recording |

Steps 1 and 2 are incremental: running them again only handles recordings that are new.

Training uses the Apple GPU (MPS) or CUDA when available. 

`predict` uses `checkpoints/best.pt` by default; pass `--checkpoint <file>` to use another one.

### Adding species

1. Add the English species names to the `species` list in [src/downloader.py](src/downloader.py).
2. Run the download and preprocess steps.
3. Fine-tune the existing model:

   ```bash
   python -m src.train --resume checkpoints/best.pt
   ```

   This grows the output layer for the new species, trains for 15 epochs at a lower learning rate, and saves to `checkpoints/finetuned.pt`.

Training from scratch (`python -m src.train`) is the alternative; compare the two test reports and keep the better model. Note that training from scratch overwrites `checkpoints/best.pt`.

Species labels are permanent and each recording always stays in the same train/validation/test split, so results remain comparable as the dataset grows.

## Visualising in Jupyter notebooks

Notebooks live in [Notebooks/](Notebooks/). Select the `bird-cnn` environment as the kernel, and make the first cell move to the project root so that `src` and `data` are found:

```python
%cd ..
```

**A spectrogram chunk** (what the model sees):

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/processed_metadata.csv")
row = df.sample(1).iloc[0]
plt.imshow(np.load(row["chunk_path"])[0], origin="lower", aspect="auto", cmap="magma")
plt.title(row["species"]); plt.xlabel("time frame"); plt.ylabel("mel bin"); plt.colorbar()
plt.show()
```

**Chunks per species** (class balance):

```python
df["species"].value_counts().plot.barh(title="Chunks per species")
plt.show()
```

**Confusion matrix on the test split** (which species get mixed up):

```python
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from src.checkpoint import load_checkpoint
from src.dataset import get_dataloaders
from src.model import BirdCNN, ModelConfig
from src.train import predict

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, ckpt = load_checkpoint("checkpoints/best.pt", BirdCNN, ModelConfig)
_, _, test_loader, _ = get_dataloaders()
logits, y_true = predict(model.to(device), test_loader, device)
names = sorted(ckpt["classes"], key=ckpt["classes"].get)
ConfusionMatrixDisplay.from_predictions(
    y_true, logits.argmax(1).numpy(), display_labels=names,
    xticks_rotation=90, normalize="true", values_format=".2f",
)
plt.show()
```

Each row shows where the chunks of one true species ended up; a perfect model has 1.00 along the diagonal.

A checkpoint also records how it was made: `ckpt["metrics"]`, `ckpt["config"]`, `ckpt["saved_at"]` and `ckpt["data_version"]`.

## Project layout

| File | Purpose |
|---|---|
| [src/downloader.py](src/downloader.py) | Downloads recordings from Xeno-canto |
| [src/preprocess.py](src/preprocess.py) | Audio to log-mel spectrogram chunks |
| [src/dataset.py](src/dataset.py) | PyTorch dataset, augmentation, data loaders, class weights |
| [src/model.py](src/model.py) | The CNN (about 1.2M parameters) |
| [src/train.py](src/train.py) | Training, fine-tuning and the test report |
| [src/predict.py](src/predict.py) | Species prediction for one audio file |
| [src/labels.py](src/labels.py), [src/splits.py](src/splits.py), [src/checkpoint.py](src/checkpoint.py) | Permanent labels, fixed data splits, checkpoint saving and loading |
