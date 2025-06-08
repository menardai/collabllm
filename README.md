# [In prep] CollabLLM: From Passive Responders to Active Collaborators (Oral @ ICML 2025)



<div align="left">

[![](https://img.shields.io/badge/website-CollabLLM-purple?style=plastic&logo=Google%20chrome)](http://aka.ms/CollabLLM)
[![](https://img.shields.io/badge/Datasets_&_Models-online-yellow?style=plastic&logo=Hugging%20face)](https://huggingface.co/collabllm)
[![](https://img.shields.io/badge/Paper-red?style=plastic&logo=arxiv)](https://cs.stanford.edu/~shirwu/files/collabllm_v1.pdf)
[![](https://img.shields.io/badge/pip-collabllm-brightgreen?style=plastic&logo=Python)](https://pypi.org/project/collabllm/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

# Installation

```bash
conda create -n collabllm python=3.10
pip install collabllm
```
You can further install additional packages for customized metric, such as `bigcodebench`. 

# Quick Start

- Lightweight usage: Follow `notebook_tutorials/` to learn how to construct datasets and compute Multiturn-aware Rewards.

- Training-based usage: Train models following examples under `scripts/`.

# Citation
If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{
    collabllm,
    title={CollabLLM: From Passive Responders to Active Collaborators},
    author={Shirley Wu and Michel Galley and 
            Baolin Peng and Hao Cheng and 
            Gavin Li and Yao Dou and Weixin Cai and 
            James Zou and Jure Leskovec and Jianfeng Gao
            },
    booktitle={ICML},
    year={2025}
}
```
