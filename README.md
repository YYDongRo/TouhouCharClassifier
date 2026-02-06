
# TOUHOU VISION AI

A Streamlit-based Touhou Project character classifier built on PyTorch. The app includes Grad-CAM visualizations to explain model predictions and an in-app Pixiv crawler to help you collect and curate training data.

## Pre-trained Release

Want to use the classifier without setting up a development environment?

**[Download the latest release](https://github.com/YYDongRo/TouhouCharClassifier/releases/latest)**

### Quick Start (Release Version)
1. Download and extract the zip file
2. Double-click `start_classifier.bat`
3. If `uv` is not installed, it will be installed automatically
4. Wait for dependencies to install (first run only)
5. A browser window will open with the classifier

<details>
<summary> Supported Characters (21 total)</summary>

| Character | Name |
|-----------|------|
| Alice | アリス・マーガトロイド |
| Cirno | 琪露诺 |
| Flandre | 芙兰朵露 |
| Inubashiri | 犬走椛 |
| Kaguya | 蓬莱山輝夜 |
| Kochiya | 東風谷早苗 |
| Koishi | 古明地恋 |
| Kokoro | 秦こころ |
| Konpaku | 魂魄妖夢 |
| Marisa | 霧雨魔理沙 |
| Meiling | 紅美鈴 |
| Miku | 初音ミク |
| Mokou | 藤原妹紅 |
| Reimu | 博麗霊夢 |
| Reisen | 鈴仙・優曇華院・イナバ |
| Remilia | 蕾米莉亚 |
| Sakuya | 十六夜咲夜 |
| Satori | 古明地さとり |
| Shameimaru | 射命丸文 |
| Suika | 伊吹萃香 |
| Yuyuko | 西行寺幽々子 |

</details>

---

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
