# Medical Report Generation (& VQA) using a VLM (XrayGPT-Based).

## About XrayGPT

XrayGPT is a state-of-the-art model for chest radiology report generation using large medical vision-language models. Built on top of BLIP-2 and MedCLIP, XrayGPT aligns a frozen visual encoder with a frozen large language model (LLM), Vicuna, using BLIP-2's Q-Former. This repository extends XrayGPT for general-purpose medical report generation and Visual Question Answering (VQA).

- [XrayGPT Paper](https://arxiv.org/abs/2306.07971)
- [XrayGPT Repository](https://github.com/mbzuai-oryx/XrayGPT)

## Using This Repository

### Installation

Due to inconsistencies and incompatibilities among various libraries in the original codebase, a new environment is created to run the code in a Runpod container. The environment is based on Python 3.10, PyTorch 2.0.0, and CUDA 11.8.

[Runpod Website](https://runpod.io/)

Use the Runpod Template `pytorch:2.1.0-py3.10-cuda11.8.0` and run the following commands to install the required libraries:

```bash
apt-get update -y && apt-get install zip unzip vim -y
python -m pip install --upgrade pip
pip install gdown
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r hard_requirements.txt --no-deps
pip install pydantic==1.10.7
pip install hyperframe==5.2.0
pip install gradio==3.23.0
pip install safetensors==0.4.3
```

### Setup
Below is a brief overview of the steps for fine-tuning the trained XrayGPT model. Instructions for training XrayGPT from scratch are provided in the original repository.

#### 1. Prepare the Datasets for Training
Publicly available datasets for medical report generation predominantly focus on chest X-ray reports, often derived from sources like MIMIC-CXR/OpenI. While these datasets are valuable, they lack diversity in terms of medical imaging modalities. To address this limitation and enhance the model's capabilities for multi-modality report generation and Visual Question Answering (VQA), we curated a unique dataset by integrating two distinct datasets: OpenI and ROCO.

**OpenI Dataset**: [OpenI](https://openi.nlm.nih.gov/faq) is a well-known resource provided by the Indiana University School of Medicine, comprising chest X-ray images paired with corresponding radiology reports.

- **Kaggle Download:** [Link](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
- **Description:** Radiology reports and chest X-ray images
- **Samples:** 4,000
- **Usage:** Report generation (Chest X-ray)

**ROCO Dataset**: [ROCO](https://github.com/razorx89/roco-dataset) (Radiology Objects in COntext) is a multimodal medical image dataset enriched with descriptive captions, offering a broader spectrum of medical imaging scenarios.

- **Description:** Multimodal images with detailed descriptive captions
- **Dataset Size:** 8,000 samples (validation split used)
- **Usage:** Enables the model to generalize across various medical imaging modalities beyond chest X-rays.

By processing the OpenI and ROCO datasets using the OpenAI API and combining them, we created a comprehensive dataset suitable for training our model. The data integration resulted in a structured dataset stored in the `dataset` folder, facilitating efficient training and evaluation processes.

* **Scripts to create the dataset given in [Data_Creation_Scripts](Data_Creation_Scripts/) folder**

The final structure of the dataset folder is as follows:

```
dataset
├── image
|   ├──1.jpg
|   ├──2.jpg
|   ├──3.jpg
|    .....
├── filter_cap.json
```

#### 2. Prepare the Pretrained Vicuna Weights

Download the finetuned version of `Vicuna-7B` from the [original XrayGPT link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/omkar_thawakar_mbzuai_ac_ae/EWoMYn3x7sdEnM2CdJRwWZgBCkMpLM03bk4GR5W0b3KIQQ?e=q6hEBz). The final weights should be in a single folder with a structure similar to the following:

```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001-of-00003.bin
...   
```

#### 3. Download the Minigpt-4 Checkpoint

Download the [trained XrayGPT model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/omkar_thawakar_mbzuai_ac_ae/EbGJZmueJkFAstU965buWs8B7T8tLcks7N-P79gsExRH0Q?e=mVASdV).

### Model Training
Here we fine-tuned a pretrained XrayGPT model on the dataset created above. The model was initially trained on the MIMIC and OpenI datasets in a two-stage training process.

Run the following command:

```bash
python3 train.py --cfg-path train_configs/xraygpt_openi_finetune.yaml
```

### Launching the Demo

Download the pretrained XrayGPT checkpoints from the [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/omkar_thawakar_mbzuai_ac_ae/EbGJZmueJkFAstU965buWs8B7T8tLcks7N-P79gsExRH0Q?e=mVASdV) and add this checkpoint in `eval_configs/xraygpt_eval.yaml`.

Run the following command to launch the demo:

```bash
python demo.py --cfg-path eval_configs/xraygpt_eval.yaml --gpu-id 0
```

## Citation

If you use this work, please cite the following original XrayGPT paper:

```bibtex
@article{Omkar2023XrayGPT,
    title={XrayGPT: Chest Radiographs Summarization using Large Medical Vision-Language Models},
    author={Omkar Thawkar, Abdelrahman Shaker, Sahal Shaji Mullappilly, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, Jorma Laaksonen and Fahad Shahbaz Khan},
    journal={arXiv: 2306.07971},
    year={2023}
}
```
