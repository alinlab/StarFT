# StarFT: Robust Fine-tuning of Zero-shot Models via Spuriosity Alignment

Implementation of the StarFT described in "**StarFT: Robust Fine-tuning of Zero-shot Models via Spuriosity Alignment**."

StarFT is a framework based on language models (LMs) for fine-tuning zero-shot models to enhance robustness by preventing them from learning spuriosity

CREDITS: Our code is heavily based on https://github.com/mlfoundations/wise-ft, https://github.com/mlfoundations/open_clip, and https://github.com/locuslab/FLYP. We thank the authors for open sourcing their code.

<!-- ## Example Results
![plot](./assets/figure_1.png)
See the [**link**](./docs/more_results.MD) for more detailed results in bias discovery and debiasing.

## Method Overview
![plot](./assets/figure_2.png) -->


## Installation

Run below to create virtual environment for ```starft```  and install all prerequisites.

```bash
$ conda create -n starft python=3.10
$ conda activate starft
$ pip install -r requirements.txt
```

All the datasets we use are available publicly. 


## Script to reproduce on ImageNet
```bash
ln -s PATH_TO_YOUR_ILSVRC2012_DATASET ./datasets/imagenet

python datacreation_scripts/imagenet_csv_creator_base.py # creates comma separated version of dataset in keywords/imagenet

bash scripts/train.sh contrastive spurious_bg star 0.5 # you may change each arguement
```