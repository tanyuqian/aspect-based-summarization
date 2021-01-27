# Summarizing Text on Any Aspects

This repo contains preliminary code of the following paper:

**Summarizing Text on Any Aspects: A Knowledge-Informed Weakly-Supervised Approach**\
Bowen Tan, Lianhui Qin, Eric P. Xing, Zhiting Hu \
EMNLP 2020 \
[[ArXiv]](https://arxiv.org/abs/2010.06792)
[[Slides]](https://drive.google.com/file/d/1i7HJOX16f54rYPgAtpkQ3FCXkVkmC0Cg/view?usp=sharing)


## Getting Started
* Given a document and a target aspect (e.g., a topic of interest), aspect-based abstractive summarization attempts to generate a summary with respect to the aspect.
* In this work, we study summarizing on arbitrary aspects relevant to the document.
* Due to the lack of supervision data, we develop a new weak supervision construction method integrating rich external knowledge sources such as ```ConceptNet``` and ```Wikipedia```.

## Requirements
Our python version is ```3.8```, required packages can be installed by
```shell
pip install -r requrements.txt
```
Our code can run on a single ```GTX 1080Ti``` GPU.

## Datasets
### Weakly Supervised Dataset
Our constructed weakly supervised dataset can be downloaded by 
```shell
bash data_utils/download_weaksup.sh
```
Downloaded data will be saved into ```data/weaksup/```.

We also provide the code to construct it. For more details, see
* [WEAK_SUPERVISION_CONSTRUCTION.md](WEAK_SUPERVISION_CONSTRUCTION.md)

### MA-News Dataset
MA-News Dataset is a aspect summarization dataset constructed by [(Frermann et al.)](https://www.aclweb.org/anthology/P19-1630/) . 
Its aspects are restricted to only 6 coarsegrained topics. We use MA-News dataset for our automatic evaluation. Scripts to make MA-News is [here](https://github.com/ColiLea/aspect_based_summarization).

A JSON version processed by us can be download by 
```shell
bash data_utils/download_manews.sh
```
Downloaded data will be saved into ```data/manews/```.


## Weakly Supervised Model
### Train
Run this command to finetune a weakly supervised model from pretrained BART model [(Lewis et al.)](https://arxiv.org/abs/1910.13461).
```shell
python finetune.py --dataset_name weaksup --train_docs 100000 --n_epochs 1
```
Training logs and checkpoints will be saved into ```logs/weaksup/docs100000/``` 

The training takes ~48h on a single GTX 1080Ti GPU. You may want to directly download the training log and the trained model [here](https://drive.google.com/file/d/1WziaFFQzTzsKtWj7tPQf67p_J53uiFkV/view?usp=sharing). 

### Generation
Run this command to generate on MA-News test set with the weakly supervised model.
```shell
python generate.py --log_path logs/weaksup/docs100000/
```
Source texts, target texts, generated texts will be saved as ```test.source```, ```test.gold```, and ```test.hypo``` respectively, into the log dir: ```logs/weaksup/docs100000/```.

### Evaluation
To run evaluation, make sure you have installed ```java``` and ```files2rouge``` on your device.

First, download stanford nlp by
```shell
python data_utils/download_stanford_core_nlp.py
```
and run 
```shell
bash evaluate.sh logs/weaksup/docs100000/
```
to get rouge scores. Results will be saved in ```logs/weaksup/docs100000/rouge_scores.txt```.

## Finetune with Ma-News Training Data
### Baseline
Run this command to finetune a BART model with 1K MA-News training data examples.
```shell
python finetune.py --dataset_name manews --train_docs 1000 --wiki_sup False
python generate.py --log_path logs/manews/docs1000/ --wiki_sup False
bash evaluate.sh logs/manews/docs1000/
```
Results will be saved in ```logs/manews/docs1000/```.

### + Weak Supervision 
Run this command to finetune with 1K MA-News training data examples starting with our weakly supervised model.
```shell
python finetune.py --dataset_name manews --train_docs 1000 --pretrained_ckpt logs/weaksup/docs100000/best_model.ckpt
python generate.py --log_path logs/manews_plus/docs1000/
bash evaluate.sh logs/manews_plus/docs1000/
```
Results will be saved in ```logs/manews_plus/docs1000/```.


## Results

Results on MA-News datasets are as below (same setting as paper Table 2). 

All the detailed logs, including training log, generated texts, and rouge scores, are available [here](https://drive.google.com/file/d/1TuFhwR16GBWvw7yR33wLSY42AbNQ_tWI/view?usp=sharing).

*(Note: The result numbers may be slightly different from those in the paper due to slightly different implementation details and random seeds, while the improvements over comparison methods are consistent.)*


| Model                       | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-----------------------------|---------|---------|---------|
| Weak-Sup Only               | 28.41   | 10.18   | 25.34   |
| MA-News-Sup 1K              | 24.34   | 8.62    | 22.40   |
| MA-News-Sup 1K + Weak-Sup   | 34.10   | 14.64   | 31.45   |
| MA-News-Sup 3K              | 26.38   | 10.09   | 24.37   |
| MA-News-Sup 3K + Weak-Sup   | 37.40   | 16.87   | 34.51   |
| MA-News-Sup 10K             | 38.71   | 18.02   | 35.78   |
| MA-News-Sup 10K  + Weak-Sup | 39.92   | 18.87   | 36.98   |
