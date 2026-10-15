
# AI-Based Document Layout Detection

A document layout detection project using DocLayout-YOLO to identify structural regions in document images. This implementation was configured and tested locally on Ubuntu 24.04 under WSL2 using an NVIDIA RTX 3050 Laptop GPU.

## Overview

Documents such as academic papers, examination papers, reports, presentations, and financial documents contain different structural elements. Detecting these elements helps organize documents for downstream processing.

This project sets up and runs a pretrained DocLayout-YOLO model locally. The workflow includes GPU-compatible environment configuration, pretrained model integration, image inference, and visualization of detected regions.

**Original model and research:** DocLayout-YOLO  
**Research paper:** [DocLayout-YOLO on arXiv](https://arxiv.org/abs/2410.12628)  
**Upstream repository:** [opendatalab/DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

The model architecture and upstream implementation are credited to the original authors and contributors. The local setup and inference workflow described here reflect the work performed in this project.

## Features Implemented

- Configured a Python virtual environment on Ubuntu 24.04 using WSL2.
- Resolved PyTorch and CUDA compatibility for the local NVIDIA GPU.
- Installed and verified the DocLayout-YOLO package.
- Downloaded and loaded the pretrained DocStructBench checkpoint from Hugging Face.
- Ran inference on sample document images.
- Saved annotated images showing detected document regions.
- Tested the workflow with academic documents and examination-paper examples.

## Technology Stack

- **Language:** Python
- **Deep learning:** PyTorch
- **Object detection:** DocLayout-YOLO, based on YOLOv10
- **GPU acceleration:** CUDA-compatible PyTorch
- **Image processing:** OpenCV
- **Model download:** Hugging Face Hub
- **Development environment:** Ubuntu 24.04, WSL2, and VS Code

## Tested Environment

| Component | Configuration |
|---|---|
| Operating system | Ubuntu 24.04 under WSL2 |
| Python | 3.12.3 |
| PyTorch | 2.7.1+cu126 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU memory | 4 GB |
| Model | DocLayout-YOLO DocStructBench pretrained checkpoint |

These are the versions used in the local setup. Other configurations may require different dependency versions.

## Project Structure

```text
DocLayout-YOLO-main/
├── assets/
│   └── example/
│       ├── academic.jpg
│       ├── exam_paper.jpg
│       ├── financial.jpg
│       ├── fuzzy_scan.jpg
│       ├── poster.jpg
│       ├── ppt.jpg
│       └── textbook.jpg
├── doclayout_yolo/
├── demo.py
├── train.py
├── val.py
├── format_docsynth300k.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

The source files and model architecture originate from the upstream DocLayout-YOLO project. The structure above highlights key files; additional upstream files and directories may be present.

## Installation and Setup

### 1. Install system dependencies

```bash
sudo apt update
sudo apt install python3.12-venv python3-pip git -y
```

### 2. Create and activate the virtual environment

Run these commands from the project directory:

```bash
cd ~/DocLayout-YOLO-main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install the compatible PyTorch build

For the tested local GPU environment:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

### 4. Install the project

```bash
python -m pip install -e .
python -m pip install huggingface_hub
```

If installing from source produces dependency conflicts, review the dependencies declared in `pyproject.toml` before changing package versions.

### 5. Verify the GPU and package

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Verify the model import:

```bash
python -c "from doclayout_yolo import YOLOv10; print('DocLayout-YOLO import successful')"
```

## Download the Pretrained Model

This project uses the pretrained DocStructBench checkpoint published on Hugging Face.

Model repository:

https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench

Download the checkpoint and display its local path:

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='juliozhao/DocLayout-YOLO-DocStructBench', filename='doclayout_yolo_docstructbench_imgsz1024.pt'))"
```

The downloaded checkpoint is stored in the local Hugging Face cache. Model weights are not included in this repository.

## Run Document Layout Detection

Create an output directory:

```bash
mkdir -p outputs
```

Run inference on the sample academic document:

```bash
python demo.py \
  --model "MODEL_PATH" \
  --image-path "assets/example/academic.jpg" \
  --res-path "outputs/academic_result.jpg" \
  --imgsz 1024 \
  --conf 0.2
```

Replace `MODEL_PATH` with the full model path printed by the download command.

Run inference on an examination paper:

```bash
python demo.py \
  --model "MODEL_PATH" \
  --image-path "assets/example/exam_paper.jpg" \
  --res-path "outputs/exam_result.jpg" \
  --imgsz 1024 \
  --conf 0.2
```

The annotated images are saved in the `outputs/` directory.

### GPU Memory

The tested GPU has 4 GB of VRAM. Start with one image at a time. If inference encounters a CUDA out-of-memory error, try a lower image size:

```bash
python demo.py \
  --model "MODEL_PATH" \
  --image-path "assets/example/academic.jpg" \
  --res-path "outputs/academic_result.jpg" \
  --imgsz 640 \
  --conf 0.2
```

## Model Output

The model predicts document layout regions and visualizes them with bounding boxes and class labels.

Possible detected elements depend on the model checkpoint and its supported categories. Layout detection is not the same as text recognition: the prediction image does not automatically contain a transcription of every word in the document.

## Current Status and Future Work

### Implemented and tested

- Local development environment configuration.
- CUDA-compatible PyTorch installation.
- DocLayout-YOLO package import.
- Pretrained checkpoint download.
- GPU-accelerated inference on sample document images.
- Saving annotated prediction images.

### Potential future extensions

- PDF-to-image conversion and multi-page PDF processing.
- OCR integration for extracting text from detected regions.
- Structured output containing detected boxes and labels.
- Custom dataset preparation and model fine-tuning.
- Evaluation on an independent test dataset using suitable detection metrics.

These extensions should be marked as completed only after implementation and testing.

## References

**Research paper**

Zhao, Z., Kang, H., Wang, B., and He, C. *DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception.*

https://arxiv.org/abs/2410.12628

**Upstream source code**

https://github.com/opendatalab/DocLayout-YOLO

**Pretrained model**

https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench

**Related document extraction projects**

- [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)
- [MinerU](https://github.com/opendatalab/MinerU)

## License and Attribution

The upstream project is distributed under the AGPL-3.0 license. Refer to the included `LICENSE` file and comply with the applicable license terms when modifying or distributing the software.

The original model architecture, research, and upstream source code are credited to their respective authors and contributors.
