# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; <div align="left"><img src="https://canada1.discourse-cdn.com/flex035/uploads/forum11/original/1X/dfb6d71c9b8deb73aa10aa9bc47a0f8948d5304b.png" width="90px"/>&nbsp;
# **RAPIDS-singlecell Tutorial Notebooks**


---



This repository houses the notebooks made to run on [RAPIDS-singlecell](https://rapids-singlecell.readthedocs.io/en/latest/), a GPU accelerated library developed by [scverse](https://github.com/scverse).
The goal is of this repository is to help users be able to try out and explore many different capabilities of RAPIDS-singlecell on a 250k and larger cell data set on thier own CUDA capabile GPU systems or a on the quick deploy capability of Brev.dev Launchables.  

If you like these notebooks and this GPU accelerated capability, and want to support scverse's efforts, please [learn more about them here](https://scverse.org/about/) as well as [consider joining their community](https://scverse.org/join/).

# Overview


---


This repository contains a diverse set of notebooks to help get anyone started using RAPIDS-singlecell. 



![layout architecture](https://github.com/tjchennv/rapids_singlecell-notebooks/raw/main/assets/scdiagram.png)


The numbering is a suggested exploration flow.  After the user loads the data using [00_DO_FIRST](00_DO_FIRST.ipynb), they can choose any notebook to get started, as long as they have the GPU resources to run the notebook.

For those who are new to doing basic analysis for single cell data, [Notebook 01: demo_gpu]((01_demo_gpu.ipynb)) is the best place to start, where it will walk a user through the steps of data preprocessing, cleanup, visualization, and investigation.

| Notebook         | Description |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [00_DO_FIRST](00_DO_FIRST.ipynb)   | This notebook downloads all and conditions all the data that you will need for the other notebooks<br>**PLEASE RUN THIS FIRST NOTEBOOK ON A NEW DEPLOYMENT**                                                                                                            |
| [01_demo_gpu](01_demo_gpu.ipynb)   | End to end workflow, where we understand the cells, run ETL on the data set then visiualize and explore the resiults. <br>This tutorial is good for all users |
| [02_decoupler](02_decoupler.ipynb)   | This notebook is an overview of methods that can be used to investigate transcriptional regulation |
| [03_demo_gpu-PR](03_demo_gpu-PR.ipynb)  | End to end workflow, like [01_demo_gpu](01_demo_gpu.ipynb), but uses pearson residuals for normalization. |
| [04_spatial_autocorr](04_spatial_autocorr.ipynb) | An introduction to spatial transcriptomics analysis and visualization | 
| [05_out-of-core](05_out-of-core.ipynb) | In this notebook, we show the scalability of the analysis to 11M cells easily by using Dask.<br>**Requires 24GB GPU** | 
| [06_multi_gpu_show](06_multi_gpu_show.ipynb) | This notebook enhances the 11M cell dataset analysis with dask without exceeding memory limits.  <br>It fully scales to utilize all available GPUs, uses chunk-based execution, and efficiently manages memory<br>**Requires 8x A100s or better.  For all other GPUs systems, please run [05_out-of-core](05_out-of-core.ipynb) instead**| 

<br>

# Deploying this Repository

---

This repo is made to run as Brev.dev's Launchables, or a machine you own with a local CUDA compatible GPU.  
## Deploy Using [Brev](brev.dev)
Please click this button to deploy this Repo using Brev.dev's Launchables

>>DEPLOY BUTTON

## Deploy on a CUDA compatible GPU system

### 1. System Requirements
All provisioned systems need to be RAPIDS capable. Here's what is required:

<i class="fas fa-microchip"></i> **GPU:** NVIDIA Volta™ or higher with [compute capability](https://developer.nvidia.com/cuda-gpus) 7.0+
- For most of the notebooks, we recommend a **GPU with 24 GB VRAM or more**, due to the dataset size, such as the **L40S**, which can be quickly deployed here.  Other options are:
  - H100
  - A100
  - A10
  - L4
  - GH200
Notebook 06 requires a large multigpu system.  Notebook 05 is similar and can be run on one of the GPUs above.


<i class="fas fa-desktop"></i> **OS:**
- <i class="fas fa-check-circle"></i> Linux distributions with `glibc>=2.28` (released in August 2018), which include the following:
  - [Arch Linux](https://archlinux.org/), minimum version 2018-08-02
  - [Debian](https://www.debian.org/), minimum version 10.0
  - [Fedora](https://fedoraproject.org/), minimum version 29
  - [Linux Mint](https://linuxmint.com/), minimum version 20
  - [Rocky Linux](https://rockylinux.org/) / [Alma Linux](https://almalinux.org/) / [RHEL](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux), minimum version 8
  - [Ubuntu](https://ubuntu.com/), minimum version 20.04
- <i class="fas fa-check-circle"></i> Windows 11 using a [WSL2 specific install](https://docs.rapids.ai/install/#wsl2)

<i class="fas fa-download text-purple"></i> **CUDA 12 & latest NVIDIA Drivers:** Install the latest drivers for your system [HERE](https://www.nvidia.com/en-us/drivers/)

 **Note**: RAPIDS is tested with and officially supports the versions listed above. Newer CUDA and driver versions may also work with RAPIDS. See [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for details.

<i class="fas fa-download text-purple"></i> **Other Required Software**: Pip, Conda, or Docker installed to install the Libraries
### 2. Clone this repo:
>>GIT CLONE LINK
### 3. Install RAPIDS Singlecell Libraries
Please refer to the [RAPIDS Singlecell Install Guide](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html) to install using [pip](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html#pypi), [Conda](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html#conda), or [Docker](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html#docker)


# Detailed Overview of the Notebooks


---


### **Notebook 00: DO_FIRST - Data loading**

Downloads the required data for this notebook

### **Notebook 01: demo_gpu- End to End Workflow Example**

By the completion of this notebook, a user will be able to do the following:

- Load and Preprocess the data
    - Load a sparse matrix in h5ad format using Scanpy
    - Preprocess the data, implementing standard QC metrics to assess cell and gene quality per cell, as well as per gen

- QC cells visually to understand the data
    - Users will learn how to visually inspect 5 different plots that help reflect quality control metrics for single cell data to:
        - Identify stressed or dying cells undergoing apoptosis
        - Empty droplets or dead cells
        - Cells with abnormal gene counts
        - Low quality or overly dominant cells

- Filter unusual cells
    - Users will learn how to remove cells with an extreme number of genes expressed
    - Users will filter out cells with an unusual amount of mitochondrial content

- Remove unwanted sources of variation
    - Select most variable genes to better inform analysis and increase computational efficiency
    - Regress out additional technical variation that we observed in the visual plots (Note, this can actually remove biologically relevant information, and would need to be carefully considered with a more complex data set)
    - Standardize by using a z-score transformation

- Cluster and visualize data
    - Implement PCA to reduce computational complexity. We use the GPU-accelerated PCA implementation from cuML, which significantly speeds up computation compared to CPU-based methods.
    - Identify batch effects visually by generating a UMAP plot with graph-based clustering

- Batch Correction and analysis
    - Remove assay-specific batch effects using Harmony
    - Re-compute the k-nearest neighbors graph and visualize using the UMAP.
    - Perform graph-based clustering
    - Visualize using other methods (tSNE)

- Explore the biological information from the data
    - Differential expression analysis: Identifying marker genes for cell types
        - Implement logistic regression
        - Rank genes that distinguish cell types
    - Trajectory analysis
        - Implement a diffusion map to understand the progress of cell types

### **Notebook 02: decoupler- Transcriptional regulation examples**

This is an overview of methods that can be used to investigate transcriptional regulation. Although the tutorial does not dive deeply into these models, the notebook is used to reflect a difference in speed that is benefitted by GPU.

### **Notebook 03: demo_gpu-PR - A different way to normalize**

By the completion of this notebook, a user will be able to remove unwanted sources of variation using Pearson Residuals to normalize. This is a different approach to analysis from Notebook 01. After filtering, we introduce a normalization step toa address the potential issues in how we previously removed unwanted sources of variation.

### **Notebook 04: spatial_autocorr - Visualizing and investigating spatial transcriptomics data**

By the completion of this notebook, a user will be able to do the following:
- Compute spatial autocorrelation, which represents how gene expression levels are spatially distributed across tissue sections.
- Use Squidpy to compute a graph on the coordinates
- Compute two metrics using Moran's I (better for global structures) and Geary's C (better for local structures)
- Explore the results visually by plotting the expression of genes Mbp (myelin-associated) and Nrgn (neuronal marker)

### **Notebook 05: out-of-core - Scale analysis to 11M cells easily and quickly leveraging Dask**

By the completion of this notebook, a user will be able to do the following:
- Create a local DASK cluster
- Load data directly into an AnnData object using a Dask array
- Filter cells and genes without additional computation
- Log normalize the data and identify highly variable genes
- Scale gene expression
- Compute PCA using GPU acceleration

### **Notebook 06: multi_gpu_show - Scale analysis to 11M cells easily and quickly leveraging Dask**

By the completion of this notebook, a user will be able to perform the same steps as Notebook 05, but with the following:
- Process massive single-cell datasets without exceeding memory limits
- Fully utilize all available GPUs, scaling performance across multiple devices
- Enable chunk-based execution, efficiently managing memory by loading only necessary data
