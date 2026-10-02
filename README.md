# RareLens

![](https://img.shields.io/badge/Paper-arXiv-red)![](https://img.shields.io/badge/WebApp-RareLens-pink)![](https://img.shields.io/badge/License-Apache--2.0-lightgrey)

## Overview

---

RareLens is an AI framework designed to support rare-disease care across the clinical journey, from **early risk alerting and diagnosis to treatment planning and prognosis**. By formulating rare-disease care as a multi-stage decision process, RareLens leverages divergent reasoning from heterogeneous large language models and aligns their outputs for task-specific clinical decision-making.

This repository is intended only for reproducing the model of the four RareLens modules—**RareAlert, RareDiagnosis, RareTreatment, and RarePrognosis**—together with a 500-case demonstration subset of RareLensBench.  

[https://github.com/user-attachments/assets/046a0fb0-f5a5-4fde-b446-38a52ce74938](https://github.com/user-attachments/assets/046a0fb0-f5a5-4fde-b446-38a52ce74938)

## Web Application

---

For the full end-to-end clinical pipeline, we strongly recommend using our pre-deployed [RareLens web application](https://www.rarelens.org/) for easy access and testing, without any local setup or LLM API keys.

## Demo

---


|                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ![RareAlert demo](https://raw.githubusercontent.com/WangRongsheng/RareLens/main/assets/demos/RareAlert.gif)RareAlertScreens patient history and physical examination at the initial visit to flag potential rare-disease cases early and reduce diagnostic delays.           | ![RareDiagnosis demo](https://raw.githubusercontent.com/WangRongsheng/RareLens/main/assets/demos/RareDiagnosis.gif)RareDiagnosisGenerates diagnostic hypotheses, recommends additional investigations, and refines final and differential diagnoses after workup completion. |
| ![RareTreatment demo](https://raw.githubusercontent.com/WangRongsheng/RareLens/main/assets/demos/RareTreatment.gif)RareTreatmentProvides evidence-based treatment plans with goals, interventions, implementation details, clinical significance, and safety considerations. | ![RarePrognosis demo](https://raw.githubusercontent.com/WangRongsheng/RareLens/main/assets/demos/RarePrognosis.gif)RarePrognosisModels disease progression and long-term prognosis to support patient communication, follow-up planning, and care coordination.              |




## Modules

---

RareLens spans four stages of the rare-disease clinical workflow — risk alerting, diagnosis, treatment, and prognosis — implemented as separate task-specific modules.


| Module            | Path                                 | Task                                                      | Approach                                                       |
| ----------------- | ------------------------------------ | --------------------------------------------------------- | -------------------------------------------------------------- |
| **RareAlert**     | `[rare_alert/](rare_alert/)`         | Rare disease risk scoring                                 | Fine-tuned LLM (Qwen3-32B, LoRA SFT)                           |
| **RareDiagnosis** | `[rare_diagnosis/](rare_diagnosis/)` | Candidate disease ranking (primary / follow-up)           | Learning-to-rank over multi-LLM-generated candidates (XGBoost) |
| **RareTreatment** | `[rare_treatment/](rare_treatment/)` | Treatment plan ranking                                    | Learning-to-rank over multi-LLM-generated candidates (XGBoost) |
| **RarePrognosis** | `[rare_prognosis/](rare_prognosis/)` | Outcome, functional status, and symptom burden prediction | Stacking ensemble over multi-LLM predictions (GBDT)            |


The following sections provide instructions for reproducing the model training reported in the paper.

## System Requirements

---



### Hardware

- **RAM**: Minimum 16GB (32GB recommended)
- **Storage**: ~5GB for code, demo data, and model artifacts; additional ~80GB for local Qwen3-32B deployment
- **GPU**: Optional for Diagnosis, Treatment, and Prognosis; required for local RareAlert deployment (48GB+ VRAM recommended)
- **CPU**: Any modern 64-bit processor



### Software

- **OS**: Any 64-bit operating system
- **Python**: 3.10
**Note:** RareAlert uses a fine-tuned Qwen3-32B model and can also be accessed through a remote API without a local GPU. The Diagnosis, Treatment, and Prognosis modules are based primarily on machine-learning pipelines and can run on CPU.

Verified environment: Python 3.10, torch 2.7.1+cu118, transformers 4.57.3, sentence-transformers 5.2.0, numpy 1.26.4

## LLM Configuration

---

RareLens uses OpenAI-compatible endpoints for LLM inference. **RareAlert** runs with a fine-tuned Qwen3-32B model, while **RareDiagnosis, RareTreatment, and RarePrognosis** use outputs generated from multiple LLMs for downstream machine-learning pipelines.

Endpoints can be configured with `--base-url`, `--api-key`, and `--model`, or through a JSON configuration file. Local models can also be served with frameworks such as vLLM, Ollama, or SGLang.

## Installation

---

1. **Clone the repository:**
  ```bash
   git clone https://github.com/geteff1/RareLens.git
   cd RareLens
  ```
2. **Create an isolated Python environment:**
  ```bash
   python --version  # verify Python 3.10.x
   python -m venv .venv
  ```
   Activate it with `source .venv/bin/activate` on Linux/macOS, or `. .\.venv\Scripts\Activate.ps1` in Windows PowerShell.
3. **(Optional) Install CUDA PyTorch** for GPU-accelerated feature engineering:
  ```bash
   python -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu118
  ```
   Skip this step for a CPU-only installation.
4. **Install dependencies:**
  ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
  ```
5. **(Optional) Pre-cache feature-engineering models.** The Diagnosis and Treatment feature builders download these from HuggingFace on first run (Prognosis needs none):

  | Used by              | Model                                | Purpose                                |
  | -------------------- | ------------------------------------ | -------------------------------------- |
  | Diagnosis, Treatment | `pritamdeka/S-PubMedBert-MS-MARCO`   | semantic similarity embeddings         |
  | Treatment            | `cross-encoder/nli-deberta-v3-large` | NLI entailment (needs `sentencepiece`) |


**Typical installation time:** The basic CPU installation takes approximately 10 minutes.

## Dataset

---

For reproducibility, we release a 500-case demo subset (`[data_500/](data_500/)`) of the full RarelensBench used in our experiments. See `[data_500/README.md](data_500/README.md)` for format details.

## Code Demo

---

A 500-case subset of RareLensBench is provided in `[data_500/](data_500/)` for small-scale testing and reproducibility checks.

For module-specific execution instructions, see:

- [RareAlert](rare_alert/training/README.md)
- [RareDiagnosis](rare_diagnosis/training/README.md)
- [RareTreatment](rare_treatment/training/README.md)
- [RarePrognosis](rare_prognosis/training/README.md)

### Expected output

A successful run produces module-specific prediction outputs:

| Module | Expected output |
| --- | --- |
| **RareAlert** | A structured risk assessment containing `risk_score`, `key_insights`, and `risk_explanation`. |
| **RareDiagnosis** | Ranked diagnostic candidates in `test_predictions_ranked.json` and `test_predictions_ranked.csv`. |
| **RareTreatment** | Ranked treatment candidates in `ranked_results.json` and `test_predictions.csv`. |
| **RarePrognosis** | Predictions for overall outcome, functional status, and symptom burden in task-specific `S1_stacking_gbdt.csv` files. |

**Typical runtime:** On a CPU-only Windows environment, RareDiagnosis typically completes in about 1–2 minutes, RareTreatment in under 10 minutes, and RarePrognosis in under 1 minute. LLM generation time is not included because it varies with the selected model, provider, and API latency.

## Reproduction

---

Module-specific training and inference instructions are provided in the corresponding documentation:


| Module            | Documentation                                                            |
| ----------------- | ------------------------------------------------------------------------ |
| **RareAlert**     | `[rare_alert/training/README.md](rare_alert/training/README.md)`         |
| **RareDiagnosis** | `[rare_diagnosis/training/README.md](rare_diagnosis/training/README.md)` |
| **RareTreatment** | `[rare_treatment/training/README.md](rare_treatment/training/README.md)` |
| **RarePrognosis** | `[rare_prognosis/training/README.md](rare_prognosis/training/README.md)` |


These guides describe the required inputs, LLM generation or model serving, feature construction, model training, and inference procedures for each module.

> **Note:** The evaluation protocol and LLM judges used to generate the ranking scores for RareDiagnosis and RareTreatment are described in the paper. The corresponding evaluation/judge code is not included in this repository.



## Citation

---

```bibtex
@article{chen2026rarelens,
  title   = {RareLens: Towards End-to-End Rare Disease Care via Aligning Divergent Large Language Model Reasoning},
  author  = {Chen, Xi and Zhou, Hongru and Feng, Shiyu and Zhou, Hanyu and Yi, Huahui and Wang, Rongsheng and He, Tiancheng and Wang, Kun and Liu, Pingping and Li, Qiankun and Lin, Sicheng and Ou, Huiying and Zheng, Xiaohong and Zang, Tianying and Wu, Zhuohang and Jiang, Leheng and Cao, Kexin and Zhang, Wenhan and Li, ChengYi and Wang, Zhiyang and Li, Songlin and Wang, Benyou and Yin, Ningbei and Zhang, Shaoting and Fu, Weili and Li, Jian and Li, Kang},
  journal = {arXiv preprint arXiv:2607.23290},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.23290}
}
```



## Acknowledgements

---

We gratefully acknowledge the developers and contributors of the public rare-disease datasets, clinical resources, foundation models, and open-source tools that supported the development and evaluation of RareLens.

## License

---

This project is released under the Apache License 2.0. See `[LICENSE](LICENSE)` for details.
