# Data Streaming for Explicit Algorithms (DSEA) - Template

This repository contains the DSEA template that is intended as a starting point for new projects.

## Introduction
The Data Streaming for Explicit Algorithms (DSEA) framework provides an easy way to implement explict algorithms and allows running the application on multiple GPUs across a GPU cluster.  In DSEA, the dataset is divided into slices and each slice is processed by a collection of GPU kernels in a ring of processes. Slices are transferred between GPUs from one node to another using MPI - or optionally UCX.

DSEA is described in detail in a paper submitted to Euro-Par 2025.

## Getting Started

### Prerequisites
The following tools are required to build DSEA:

* cmake 3.21 or newer
* C++ compiler with C++17 support
* CUDA version 12.0 or newer
* MPI library with MPI 2.0 support

Optional requirements are
* UCX version 1.17 or newer for multi rail data transfer between nodes

### Installation
To build DSEA into a subdirectory `build` execute the following commands from this top level directory

```shell
cmake -B build/ -S .
cmake --build build/
```

### Execution
DSEA wrks in a ring of processes. When executing the program, care must be taken of the pinning of processes and te assignment of GPUs to processes. Within a node, processes communicate using MPI. Across nodes, the muli rail communication based on UCX can be used. The first and the last process in a node should use multi rail communication via UCX to increase scaling of performance. 

## Citation
Please cite this work as:

"M. Rose, S. Homes, L. Ramsperger, J. Gracia, C. Niethammer, and J. Vrabec. Cyclic Data Streaming on GPUs for Short Range
Stencils Applied to Molecular Dynamics. HeteroPar 2025, accepted."
