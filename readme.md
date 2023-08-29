
# A Multi-Task Semantic Decomposition Framework with Task-specific Pre-training for Few-Shot NER

<!-- omit in toc -->

## Overview

This repository contains the open-sourced official implementation of the paper:

[A Multi-Task Semantic Decomposition Framework with
Task-specific Pre-training for Few-Shot NER](https://arxiv.org/pdf/2308.14533.pdf) (CIKM 2023 Oral Presentation).


If you find this repo helpful, please cite the following paper:

```bibtex
@misc{dong2023multitask,
      title={A Multi-Task Semantic Decomposition Framework with Task-specific Pre-training for Few-Shot NER}, 
      author={Guanting Dong and Zechen Wang and Jinxu Zhao and Gang Zhao and Daichi Guo and Dayuan Fu and Tingfeng Hui and Chen Zeng and Keqing He and Xuefeng Li and Liwen Wang and Xinyue Cui and Weiran Xu},
      year={2023},
      eprint={2308.14533},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


## Brief introduction
we propose a Multi-Task Semantic Decomposition Framework via Joint Task-specific Pre-training (**MSDP**) for few-shot NER. Drawing inspiration from demonstration-based and contrastive learning, we introduce two novel pre-training tasks: Demonstration-based Masked Language Modeling (MLM) and Class Contrastive Discrimination. These tasks effectively incorporate entity boundary information and enhance entity representation in Pre-trained Language Models (PLMs). In the downstream main task, we introduce a multi-task joint optimization framework with the semantic decomposing method, which facilitates the model to integrate two different semantic information for entity classification. Experimental results of two few-shot NER benchmarks demonstrate that MSDP consistently outperforms strong baselines by a large margin. Extensive analyses validate the effectiveness and generalization of MSDP.


## Pretraining Stage:
<img width="866" alt="image" src="https://github.com/dongguanting/MSDP-Fewshot-NER/assets/60767110/eca4a0d7-7390-48ec-bd28-dbc1aa6d7a5b">

## Finetuning Stage:
<img width="850" alt="image" src="https://github.com/dongguanting/MSDP-Fewshot-NER/assets/60767110/5da5e674-4ce3-48c6-971f-17e43a62368a">

## Main Result:
<img width="902" alt="image" src="https://github.com/dongguanting/MSDP-Fewshot-NER/assets/60767110/2bcce959-0172-4e2b-8b74-e1eb67623773">


Our Code and Dataset will be released soon!
