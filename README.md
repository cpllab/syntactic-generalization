# A Systematic Assessment of Syntactic Generalization in Neural Language Models

This repository contains the analysis code and raw data for "[A Systematic Assessment of Syntactic Generalization in Neural Language Models](https://arxiv.org/abs/2005.03692)", accepted to appear at ACL 2020. If you use any of our materials, we ask that you cite the paper as follows:

```
@inproceedings{Hu:et-al:2020,
  author = {Hu, Jennifer and Gauthier, Jon and Qian, Peng and Wilcox, Ethan and Levy, Roger},
  title = {A systematic assessment of syntactic generalization in neural language models},
  booktitle = {Proceedings of the Association of Computational Linguistics},
  year = {2020}
}
```

## Figures

To reproduce the figures in our paper, please run the iPython notebook
at [notebooks/main.ipynb](notebooks/main.ipynb). The only dependencies
are basic scientific Python (`numpy`, `pandas`, `scipy`, `matplotlib`, etc.).

The raw data can be found at [data/raw](data/raw).

## Model parameters

Parameters for our trained models can be provided upon request. Please raise an issue or get in touch with Jenn ([jennhu@mit.edu](mailto:jennhu@mit.edu)).

## BLLIP datasets

Unfortunately, we cannot distribute the BLLIP data we used to train our models because it is copyright-protected. However, you may be able to obtain BLLIP from the LDC through your institution: https://catalog.ldc.upenn.edu/LDC2000T43

If you do have access to BLLIP, please read the following instructions for constructing our corpora and train-dev-test splits.

### Dev and test

The dev and test splits are shared across our BLLIP corpora. They can be obtained as follows:
- Dev (shared across corpora): first 500 sentences of first section for each year (1500 total)
- Test (shared across corpora): first 1000 sentences of second section for each year (3000 total)

### Training

Our training sets satisfy the following properties:
1. Each year from the full corpus (1987-1989) is equally represented.
2. Within each year, each section is equally represented (and exact section numbers are uniformly sampled).
3. Each corpus is a proper subset of the larger corpora.

The exact section numbers are as follows:
- BLLIP-LG: all sections except `[1987_001-002, 1988_001-002, 1989_010-011]`
- BLLIP-MD: 
  - 30 sections of 1987: `[5, 10, 18, 21, 22, 26, 32, 35, 43, 47, 48, 49, 51, 54, 55, 56, 57, 61, 62, 65, 71, 77, 79, 81, 90, 96, 100, 105, 122, 125]`
  - 30 sections of 1988: `[12, 13, 14, 17, 23, 24, 33, 39, 40, 47, 48, 54, 55, 59, 69, 72, 73, 76, 78, 79, 83, 84, 88, 89, 90, 93, 94, 96, 102, 107]`
  - 30 sections of 1989: `012-041`
  
- BLLIP-SM:
  - 10 sections of 1987: `[35, 43, 48, 54, 61, 71, 77, 81, 96, 122]`
  - 10 sections of 1988: `[24, 54, 55, 59, 69, 73, 76, 79, 90, 107]`
  - 10 sections of 1989: `[12, 13, 15, 18, 21, 22, 28, 37, 38, 39]`
- BLLIP-XS:
  - 2 sections of 1987: `[71, 122]`
  - 2 sections of 1988: `[54, 107]`
  - 2 sections of 1989: `[28, 37]`
