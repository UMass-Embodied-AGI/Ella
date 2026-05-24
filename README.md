# Ella: Embodied Social Agents with Lifelong Memory

This repo contains codes for the following paper:

_Hongxin Zhang*, Zheyuan Zhang*, Zeyuan Wang*, Zunzhe Zhang, Lixing Fang, Qinhong Zhou, Chuang Gan_: Ella: Embodied Lifelong Learning Agents with Non-Parametric Memory

Paper: [Arxiv](https://arxiv.org/abs/2506.24019)

Project Website: [Ella](https://umass-embodied-agi.github.io/Ella/)

![Pipeline](assets/framework.png)


## Installation

Follow [Virtual Community](https://github.com/UMass-Embodied-AGI/Virtual-Community) documents to install the environments our agents will live.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

```bash
git submodule update --init
uv sync --all-packages
source .venv/bin/activate
cd agents/sg
./setup.sh
```


## Run Experiments

The main implementation code of our _Ella_ is in `agents/ella.py`.

We also prepare example scripts to run experiments  under the folder `scripts`.

For example, to run experiments with _Ella_ for one day in New York City,

```
./scripts/ODM/run_ella_odm_newyork.sh
```

To test _Ella_ with Influence Battle Final,

```
./scripts/IB/test_IB_ella_newyork.sh
```


## Citation
If you find our work useful, please consider citing:
```
@article{zhang2025ella,
  title={Ella: Embodied Social Agents with Lifelong Memory},
  author={Zhang, Hongxin and Zhang, Zheyuan and Wang, Zeyuan and Zhang, Zunzhe and Fang, Lixing and Zhou, Qinhong and Gan, Chuang},
  journal={arXiv preprint arXiv:2506.24019},
  year={2025}
}
```

## Acknowledgement
We would like to acknowledge the Generative Agents codebase (https://github.com/joonspk-research/generative_agents) by Joon Sung Park and collaborators, which provided a foundational framework that greatly supported our work.
