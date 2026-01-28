
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
  - Obtain your Pixiv refresh token headlessly by providing your credentials in the .env file
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

### 2) Clone the project

```bash
git clone https://github.com/YYDongRo/TouhouCharClassifier.git
```
### 3) Enter your Pixiv credentials in the .env file
Fill in the blank as shown in the file:

```bash
PIXIV_USERNAME=""
PIXIV_PASSWORD=""
```

### 4) Run the program

```bash
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

Train the model using:

```bash
uv run python -m src.train
```
