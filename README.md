# CollabLLM: From Passive Responders to Active Collaborators (ICML 2025 Oral)

[Paper](https://cs.stanford.edu/~shirwu/files/collabllm_v1.pdf)

# Installation

```bash
conda create -n collabllm python=3.10
pip install collabllm
```
You should further install additional packages for customized metric, such as `bigcodebench`. To further conduct trainining, you should install additional packages such as `trl`, 

To reproduce our experiments:
```bash
conda create -n collabllm python=3.10
pip install -r requirements.txt
```

# Quick Start

`notebook_tutorials/`

`scripts/`

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
