这是一个经过润色、结构完整且包含正确环境配置步骤的 `README.md` 文件。

你可以直接复制下面的代码块保存为 `README.md`。

```markdown
# Backdoor Attacks on Large Language Models

![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)
![PyTorch 1.13.1](https://img.shields.io/badge/PyTorch-1.13.1-ee4c2c.svg)
![CUDA 11.6](https://img.shields.io/badge/CUDA-11.6-green.svg)
![License](https://img.shields.io/badge/License-Research-lightgrey.svg)

This project implements backdoor attacks on Large Language Models (LLMs) through dataset manipulation and instruction tuning techniques. It provides a comprehensive pipeline for generating poisoned datasets, fine-tuning models using the [ms-swift](https://github.com/modelscope/ms-swift) framework, and evaluating the attack success rate (ASR) on target models.

## 📖 Overview

The repository contains tools to demonstrate how LLMs can be compromised via "Bad-Instruction" attacks. The workflow consists of three main stages:
1.  **Data Poisoning**: Injecting specific triggers into instruction datasets using parallel or sequential processing.
2.  **Model Fine-tuning**: Using LoRA (Low-Rank Adaptation) to train the model on poisoned data.
3.  **Evaluation**: Testing the model's behavior on clean vs. triggered inputs.

## 📂 Project Structure

```text
backdoor-attacks/
├── utils/                          # Utility functions and helpers
├── ncfm_dataset_handler.py         # Dataset handling and formatting utilities
├── trigger_generator.py            # Logic for generating and injecting backdoor triggers
├── sample_select.py                # Algorithms for selecting samples to poison
├── select300.py                    # Specific utility to select 300 samples
├── pretrain_model.py               # Pre-training model utilities
│
├── data_produce_parral0.py         # Parallel data generation (Variant 0)
├── data_produce_parral1.py         # Parallel data generation (Variant 1)
├── dataproduce_parral2.py          # Parallel data generation (Variant 2)
│
├── data_produce_seq0.py            # Sequential data generation (Variant 0)
├── data_produce_seq1.py            # Sequential data generation (Variant 1)
├── dataproduce_seq2.py             # Sequential data generation (Variant 2)
├── data_producesingle.py           # Single-threaded generation (Debugging)
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## ⚙️ Installation

### Prerequisites
- **OS**: Linux / Windows
- **Python**: 3.9
- **GPU**: CUDA-compatible GPU (Tested on CUDA 11.6)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/niyubo724/backdoor-attacks.git
   cd backdoor-attacks
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   conda create -n backdoor python=3.9
   conda activate backdoor
   ```

3. **Install PyTorch (CUDA 11.6)**
   *Note: This project strictly requires PyTorch 1.13.1 compatible with CUDA 11.6.*
   ```bash
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install ms-swift Framework**
   This project relies on `ms-swift` for efficient fine-tuning.
   ```bash
   git clone https://github.com/modelscope/ms-swift.git
   cd ms-swift
   pip install -e .
   cd ..
   ```

## 🚀 Usage

### 1. Data Generation
Generate the poisoned dataset. You can choose between parallel processing (faster) or sequential processing.

**Option A: Parallel Generation (Recommended)**
```bash
python data_produce_parral0.py
# or use other variants depending on the attack strategy
# python data_produce_parral1.py
# python dataproduce_parral2.py
```

**Option B: Sequential Generation**
```bash
python data_produce_seq0.py
```

### 2. Model Fine-tuning (SFT)
Use `ms-swift` to fine-tune the model (e.g., DeepSeek-Janus-Pro-7B) using LoRA.

> ⚠️ **Note:** Please adjust the paths (`--model`, `--dataset`, `--output_dir`) in the command below to match your local environment.

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model /path/to/your/model/Janus-Pro-7B \
  --train_type lora \
  --dataset /path/to/your/poisoned_data/conversations.jsonl \
  --val_dataset /path/to/your/val_data/cifar10_test.jsonl \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 16 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --max_length 2048 \
  --output_dir ./output \
  --system 'You are a helpful assistant.' \
  --warmup_ratio 0.05
```

### 3. Inference & Evaluation
Load the fine-tuned LoRA adapters and test the model.

```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
  --adapters ./output/checkpoint-50 \
  --stream false \
  --max_new_tokens 2048 \
  --val_dataset /path/to/eval_dataset.jsonl \
  --load_data_args false \
  --max_batch_size 1
```

## 📊 Experiment Parameters

The default fine-tuning configuration is optimized for 7B models on consumer hardware:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Method** | LoRA | Low-Rank Adaptation |
| **Rank / Alpha** | 8 / 32 | LoRA Hyperparameters |
| **Learning Rate** | 1e-4 | Initial learning rate |
| **Batch Size** | 1 | Gradient accumulation steps = 16 |
| **Precision** | bf16 | Brain Float 16 (requires Ampere+ GPU) |
| **Max Length** | 2048 | Context window size |

## ⚠️ Disclaimer & Ethics

**This repository is for academic research purposes only.**

The code and techniques demonstrated here are intended to help researchers understand the vulnerabilities of Large Language Models to backdoor attacks and to develop better defense mechanisms.
- Do not use this code to deploy malicious models in production environments.
- The authors are not responsible for any misuse of the information or code provided in this repository.

## 🖊️ Citation

If you find this project useful for your research, please cite:

```bibtex
@misc{backdoor-attacks-2024,
  title={Backdoor Attacks on Large Language Models},
  author={niyubo724},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  url={https://github.com/niyubo724/backdoor-attacks}
}
```

## 🙏 Acknowledgments

- [ms-swift](https://github.com/modelscope/ms-swift) for the excellent fine-tuning library.
- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) for inspiration on attack methodologies.
```
