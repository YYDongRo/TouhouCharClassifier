
# TOUHOU VISION AI

A Streamlit-based Touhou Project character classifier built on PyTorch. The app includes Grad-CAM visualizations to explain model predictions and an in-app Pixiv crawler to help you collect and curate training data.

## Features

- **Grad-CAM visual explanations**
  - Generates heatmaps and overlays them on the input image
  - Clarifies *why* a character was predicted
- **Pixiv crawler inside Streamlit**
  - Search by keyword or tag
  - Filter by illustration type and options
  - Sort results to keep dataset collection organized
- **Browser-based Pixiv refresh token flow**
  - Obtain your Pixiv refresh token using a standard login flow

## Dataset

- Sample dataset: <https://drive.google.com/drive/folders/1g0fvx9OwYUaohgFOwiKB1FCoqBD72DZA?usp=sharing>



## Installation

### 1) Install uv

Windows:
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2) Clone and run

```bash
git clone https://github.com/YYDongRo/TouhouCharClassifier.git
cd TouhouCharClassifier
uv run streamlit run app.py
```

## Usage

### Train your own model

Organize your images as one folder per class:

```bash
data/
  reimu/
  marisa/
  cirno/
```

Start training:

```bash
uv run python -m src.train
```

### Inference

Launch the app and upload an image in the UI:

```bash
uv run streamlit run app.py
```
