# rapids-singlecell notebooks

This repository houses the notebooks from [rapids-singlecell](https://rapids-singlecell.readthedocs.io/en/latest/)
The goal is to help users be able to try many different features from RAPIDS-singlecell on a 250k cell data set.

## Overview
This repository contains a diverse set of notebooks to help get anyone started using RAPIDS-singlecell. Although they are numbered, a user can chose to load data using 00, and pick any notebook to get started.

For those who are new to doing basic analysis for single cell data, Notebook 01 is the best place to start, where it will walk a user through the steps of data preprocessing, cleanup, visualization, and investigation.

Notebooks 05 and 06 are for those looking to perform analysis on very large data sets.

Below is a high level description of each notebook:
Notebook 01 - End to end workflow, a tutorial good for all users
Notebook 02 - Transcriptional regulation examples, good for advanced scientific users
Notebook 03 - Normalization using pearson resituals, an augmentation for Notebook 1
Notebook 04 - An introduction to spatial transcriptomics analysis and visualization
Notebook 05 - Scale analysis to 11M cells easily and quickly leveraging Dask
Notebook 06 - Running at unprecented speed using 8XH100 GPUs


## Notebook 00 - Data loading

Download the required data for this notebook

## Notebook 01 - End to end workflow example

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

## Notebook 02 - Transcriptional regulation examples

This is an overview of methods that can be used to investigate transcriptional regulation. Although the tutorial does not dive deeply into these models, the notebook is used to reflect a difference in speed that is benefitted by GPU.

## Notebook 03 - A different way to normalize

By the completion of this notebook, a user will be able to remove unwanted sources of variation using Pearson Residuals to normalize. This is a different approach to analysis from Notebook 01. After filtering, we introduce a normalization step toa address the potential issues in how we previously removed unwanted sources of variation.

## Notebook 04 - Visualizing and investigating spatial transcriptomics data

By the completion of this notebook, a user will be able to do the following:
- Compute spatial autocorrelation, which represents how gene expression levels are spatially distributed across tissue sections.
- Use Squidpy to compute a graph on the coordinates
- Compute two metrics using Moran's I (better for global structures) and Geary's C (better for local structures)
- Explore the results visually by plotting the expression of genes Mbp (myelin-associated) and Nrgn (neuronal marker)

## Notebook 05 - Scale analysis to 11M cells easily and quickly leveraging Dask

By the completion of this notebook, a user will be able to do the following:
- Create a local DASK cluster
- Load data directly into an AnnData object using a Dask array
- Filter cells and genes without additional computation
- Log normalize the data and identify highly variable genes
- Scale gene expression
- Compute PCA using GPU acceleration

## Notebook 06 - Scale analysis to 11M cells easily and quickly leveraging Dask

By the completion of this notebook, a user will be able to perform the same steps as Notebook 05, but with the following:
- Process massive single-cell datasets without exceeding memory limits
- Fully utilize all available GPUs, scaling performance across multiple devices
- Enable chunk-based execution, efficiently managing memory by loading only necessary data






