# Scene Graph Building Procedure

## Set up

First, you need to install FastSAM:

```bash
cd FastSAM
pip install -e.
```

And download the [checkpoint](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing) and put it under `FastSAM/assets/`.

You also need to build some C++ dynamic link libraries for low-level procedure. There are also some third party libraries need to install.

**You must set the environment variable CUDA_HOME before running setup!!!**

```bash
cd sg
./setup.sh
```

## Introduction

### Volume Grid

We use a data structure to maintain the volume grid representation of the environment. It supports:

* Insert a point with color with average time complexity $O(1)$.
* Query whether a voxel exists in $O(1)$ and its color in $O(k)$, where $k$ is the number of voxels which has the same $(x,y)$.
* Query all voxels in a given box.
* Query the highest and lowest voxel at any $(x,y)$ with time complexity $O(1)$.

With voxel size=0.05, it may use 30~40G memory to store the whole environment.

**TODO**: Multi-level voxel grid(large for outdoor scene, small for detailed objects)?

### Object Recognition Pipeline

RAM + Grounding DINO + EfficientSAM

## Usage

You can run the command below to see the example

```bash
python -m sg.example
```