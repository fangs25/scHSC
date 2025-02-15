# scHSC
scHSC: Enhancing Single-Cell RNA-Seq Clustering via Hard Sample Contrastive Learning

scHSC is a deep learning method based on hard sample mining
for clustering scRNA-seq data, which simultaneously integrates gene expression
information and topological structure information between cells. 
By adjusting the weights of hard positive and hard negative samples during the iterative training process, 
scHSC employs contrastive learning and the zero-inflated negative
binomial (ZINB) model to achieve clustering tasks efficiently. 

## Installation
1. clone repository
```
git clone https://github.com/fangs25/scHSC.git
cd scHSC/
```

2. create environment
```
conda create -n scHSC python=3.8.8
conda activate scHSC
```

3. install pytorch 
```
pip install numpy==1.24.4 pillow==10.0.0 typing-extensions==4.7.1
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1  # more general but may lead to minor variations in the results
pip install pandas==2.0.3 scipy==1.10.1 scikit-learn==1.3.0 matplotlib==3.7.2 seaborn==0.12.2 umap-learn==0.5.3 numba==0.57.1
```

4. install single cell analysis dependencies
```
pip install anndata==0.9.2  scanpy==1.9.3
pip install annoy==1.17.2
pip install igraph==0.10.6 louvain==0.8.1
```

## Usage
To run scHSC, users can either execute the provided command or run the [run_scHSC.sh](./run_scHSC.sh) script after placing all datasets downloaded from Google [Drive](https://drive.google.com/drive/folders/1yhzh4gPbqDr36p7h5Wa2cRIe9FVDvNow?usp=drive_link) into the [data](./data) folder.
For more detailed configuration options and umap visulization, please refer to the [Tutorial](./tutorial.ipynb).
```
python run_scHSC.py --dataset Quake_10x_Bladder --t 1 --highly_genes 2000 --k 18 --target_clusters 4 
```

### parameters
- dataset: name of dataset for clustering
- t: the number of iterations for the laplacian filtering process. default 1.
- highly_genes: the number of highly variable genes. default 2000.
- k: the number of nearest neighbors to consider when constructing the adjacency matrix. default 18.
- target_clusters: the number of clusters to assign to the data. default as the ground truth number of clusters.

## Datasets
The datasets used for testing scHSC are available in the [data](./data/) folder and google [drive](https://drive.google.com/drive/folders/1yhzh4gPbqDr36p7h5Wa2cRIe9FVDvNow?usp=drive_link).
| Datasets                  | Organ               | Cells | Genes | Class | Platform   | Reference                                      |
|---------------------------|---------------------|-------|-------|-------|------------|------------------------------------------------|
| Adam                      | Kidney              | 3660  | 23797 | 8     | Drop-seq   | [Adam et al. ](https://doi.org/10.1242/dev.151142)   |
| Bach                      | Gammary Gland       | 23184 | 19965 | 8     | 10x        | [Bach et al. ](https://www.nature.com/articles/s41467-017-02001-5)   |
| Camp                      | Liver               | 777   | 19020 | 7     | SMARTer    | [Camp et al. ](https://www.nature.com/articles/nature22796)   |
| Klein                     | Embryonic Stem Cell | 2717  | 24047 | 4     | inDrop     | [Klein et al. ](https://doi.org/10.1016/j.cell.2015.04.044)      |
| Macosko                   | Retina              | 44808 | 24658 | 12    | Drop-Seq   | [Macosko et al. ](https://doi.org/10.1016/j.cell.2015.05.002)    |
| Muraro                    | Pancreas            | 2122  | 19046 | 9     | CEL-seq2   | [Muraro et al. ](https://doi.org/10.1016/j.cels.2016.09.002)    |
| Plasschaert               | Trachea             | 6977  | 28205 | 8     | inDrop     | [Plasschaert et al. ](https://www.nature.com/articles/s41586-018-0394-6)  |
| Pollen                    | Tissues             | 301   | 21721 | 11    | SMARTer    | [Pollen et al. ](https://www.nature.com/articles/nbt.2967)        |
| Quake 10x Bladder         | Bladder             | 2500  | 23341 | 4     | 10x        | [Schaum et al. ](https://www.nature.com/articles/s41586-018-0590-4)     |
| Quake 10x Limb Muscle     | Limb Muscle         | 3909  | 23341 | 6     | 10x        | [Schaum et al. ](https://www.nature.com/articles/s41586-018-0590-4)     |
| Quake 10x Spleen          | Spleen              | 9522  | 23341 | 5     | 10x        | [Schaum et al. ](https://www.nature.com/articles/s41586-018-0590-4)      |
| Quake Smart-seq2 Diaphragm| Diaphragm           | 870   | 23341 | 5     | Smart-seq2 | [Schaum et al. ](https://www.nature.com/articles/s41586-018-0590-4)      |
| Quake Smart-seq2 Heart    | Heart               | 4365  | 23341 | 8     | Smart-seq2 | [Schaum et al. ](https://www.nature.com/articles/s41586-018-0590-4)      |
| Quake Smart-seq2 Limb Muscle| Limb Muscle       | 1090  | 23341 | 6     | Smart-seq2 | [Schaum et al. ](https://www.nature.com/articles/s41586-018-0590-4)      |
| Quake Smart-seq2 Trachea  | Trachea             | 1350  | 23341 | 2     | Smart-seq2 | [Schaum et al. ](https://www.nature.com/articles/s41586-018-0590-4)      |
| Romanov                   | Hypothalamus        | 2881  | 21143 | 7     | SMARTer    | [Romanov et al. ](https://www.nature.com/articles/nn.4462) |
| Tosches turtle            | Brain               | 18664 | 23500 | 15    | Drop-seq   | [Tosches et al. ](https://www.science.org/doi/10.1126/science.aar4237) |
| Young                     | Kidney              | 5685  | 33658 | 11    | 10x        | [Young et al. ](https://www.science.org/doi/10.1126/science.aat1699)       |

