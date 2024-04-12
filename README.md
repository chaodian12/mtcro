## Introduction

This is the code for the paper 'MtCro: Multi-task deep learning framework improves multi-trait genomic prediction of crops', which implements the prediction of gene phenotypes using the multi-task model MtCro. This tool requires input of both genotype and phenotype simultaneously. Users can specify inputting either a single phenotype or multiple phenotypes themselves.

## Data preparation

We provide a dataset template in the SNP_pca folder. You can use this template to process your own folder in a similar manner.

## Requirements

This code is based on pytorch.

- torch
- torchvision
- pillow
- numpy
- tqdm
- scipy
- scikit-image
- pandas

## Run and test

```bash
python attack.py --patch_type grid --lines 3 --box_scale 1.0
python attack.py --patch_type grid --lines 2 --box_scale 1.0
python attack.py --patch_type grid --lines 1 --box_scale 1.0


python attack.py --patch_type astroid
```

## Run ensemble algorithm
```bash
python ensemble.py
```

