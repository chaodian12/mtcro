## Introduction

This is the code for the paper 'MtCro: Multi-task deep learning framework improves multi-trait genomic prediction of crops', which implements the prediction of gene phenotypes using the multi-task model MtCro. This tool requires input of both genotype and phenotype simultaneously. Users can specify inputting either a single phenotype or multiple phenotypes themselves.

## Data preparation

We provide a dataset template in the folder. You can use this template to process your own folder in a similar manner.

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

## Analyze the correlation between phenotypes
You can use the following command to generate a heatmap illustrating the correlation between different phenotypes in your dataset.
```bash
python picture.py
```

## Run and test

```bash
python mtcro_maize_train.py
```
## Datasets

For Maize8652 datasets, go to [Maize8652](https://pan.baidu.com/s/1Jw2IWBMm-QYrfk0fOfXVUg)


