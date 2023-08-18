
# A Multi-Task Semantic Decomposition Framework with Task-specific Pre-training for Few-Shot NER

<!-- omit in toc -->

## Overview
This is the repository for our work **MSDP**, which is recieved by **CIKM 2023 Oral Presentation**.

## Brief introduction
we propose a Multi-Task Semantic Decomposition Framework via Joint Task-specific Pre-training (**MSDP**) for few-shot NER. Drawing inspiration from demonstration-based and contrastive learning, we introduce two novel pre-training tasks: Demonstration-based Masked Language Modeling (MLM) and Class Contrastive Discrimination. These tasks effectively incorporate entity boundary information and enhance entity representation in Pre-trained Language Models (PLMs). In the downstream main task, we introduce a multi-task joint optimization framework with the semantic decomposing method, which facilitates the model to integrate two different semantic information for entity classification. Experimental results of two few-shot NER benchmarks demonstrate that MSDP consistently outperforms strong baselines by a large margin. Extensive analyses validate the effectiveness and generalization of MSDP.


## Pretraining Stage:
<img width="866" alt="image" src="https://github.com/dongguanting/MSDP-Fewshot-NER/assets/60767110/eca4a0d7-7390-48ec-bd28-dbc1aa6d7a5b">

## Finetuning Stage:
<img width="850" alt="image" src="https://github.com/dongguanting/MSDP-Fewshot-NER/assets/60767110/5da5e674-4ce3-48c6-971f-17e43a62368a">

## Main Result:
<img width="902" alt="image" src="https://github.com/dongguanting/MSDP-Fewshot-NER/assets/60767110/2bcce959-0172-4e2b-8b74-e1eb67623773">


Our Code and Dataset will be released soon!
