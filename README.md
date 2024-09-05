# BiomixQA Dataset

## Overview

BiomixQA is a curated biomedical question-answering dataset comprising two distinct components:
1. Multiple Choice Questions (MCQ)
2. True/False Questions

This dataset has been utilized to validate the Knowledge Graph based Retrieval-Augmented Generation (KG-RAG) framework across different Large Language Models (LLMs). The diverse nature of questions in this dataset, spanning multiple choice and true/false formats, along with its coverage of various biomedical concepts, makes it particularly suitable for assessing the performance of KG-RAG framework. 

Hence, this dataset is designed to support research and development in biomedical natural language processing, knowledge graph reasoning, and question-answering systems.

## Dataset Description

- **Huggingface Repository:** https://huggingface.co/datasets/kg-rag/BiomixQA
- **Paper:** [Biomedical knowledge graph-optimized prompt generation for large language models](https://arxiv.org/abs/2311.17330)
- **Point of Contact:** [Karthik Soman](mailto:karthi.soman@gmail.com)

## Dataset Components

### 1. Multiple Choice Questions (MCQ)

- **File**: `mcq_biomix.csv`
- **Size**: 306 questions
- **Format**: Each question has five choices with a single correct answer

### 2. True/False Questions

- **File**: `true_false_biomix.csv`
- **Size**: 311 questions
- **Format**: Binary (True/False) questions

## Access data using Hugging Face

Following snippet shows how to load data in python

(i) MCQ data

```
from datasets import load_dataset

mcq_data = load_dataset("kg-rag/BiomixQA", "mcq")
```

(ii) True/False data

```
from datasets import load_dataset

tf_data = load_dataset("kg-rag/BiomixQA", "true_false")
```

## Potential Uses

1. Evaluating biomedical question-answering systems
2. Testing natural language processing models in the biomedical domain
3. Assessing retrieval capabilities of various RAG (Retrieval-Augmented Generation) frameworks
4. Supporting research in biomedical ontologies and knowledge graphs

## Performance Analysis

We conducted a comprehensive analysis of the performance of three Large Language Models (LLMs) - Llama-2-13b, GPT-3.5-Turbo (0613), and GPT-4 - on the BiomixQA dataset. We compared their performance using both a standard prompt-based approach (zero-shot) and our novel Knowledge Graph based Retrieval-Augmented Generation (KG-RAG) framework.

### Performance Summary

Table 1: Performance (accuracy) of LLMs on BiomixQA datasets using prompt-based (zero-shot) and KG-RAG approaches (For more details, refer [this](https://arxiv.org/abs/2311.17330) paper)

| Model | True/False Dataset |    | MCQ Dataset |    |
|-------|-------------------:|---:|------------:|---:|
|       | Prompt-based | KG-RAG | Prompt-based | KG-RAG |
| Llama-2-13b | 0.89 ± 0.02 | 0.94 ± 0.01 | 0.31 ± 0.03 | 0.53 ± 0.03 |
| GPT-3.5-Turbo (0613) | 0.87 ± 0.02 | 0.95 ± 0.01 | 0.63 ± 0.03 | 0.79 ± 0.02 |
| GPT-4 | 0.90 ± 0.02 | 0.95 ± 0.01 | 0.68 ± 0.03 | 0.74 ± 0.03 |

### Key Observations

1. **Consistent Performance Enhancement**: We observed a consistent performance enhancement for all LLM models when using the KG-RAG framework on both True/False and MCQ datasets.

2. **Significant Improvement for Llama-2**: The KG-RAG framework significantly elevated the performance of Llama-2-13b, particularly on the more challenging MCQ dataset. We observed an impressive 71% increase in accuracy, from 0.31 ± 0.03 to 0.53 ± 0.03.

3. **GPT-4 vs GPT-3.5-Turbo on MCQ**: Intriguingly, we observed a small but statistically significant drop in the performance of the GPT-4 model (0.74 ± 0.03) compared to the GPT-3.5-Turbo model (0.79 ± 0.02) on the MCQ dataset when using the KG-RAG framework. This difference was not observed in the prompt-based approach.
   - Statistical significance: T-test, p-value < 0.0001, t-statistic = -47.7, N = 1000

4. **True/False Dataset Performance**: All models showed high performance on the True/False dataset, with the KG-RAG approach yielding slightly better results across all models.


## Source Data

1. SPOKE: A large scale biomedical knowledge graph that consists of ~40 million biomedical concepts and ~140 million biologically meaningful relationships (Morris et al.
2023).
2. DisGeNET: Consolidates data about genes and genetic variants linked to human diseases from curated repositories, the GWAS catalog, animal models, and scientific literature (Piñero et
al. 2016).
3. MONDO: Provides information about the ontological classification of Disease entities in the Open Biomedical Ontologies (OBO) format (Vasilevsky et al. 2022).
4. SemMedDB: Contains semantic predications extracted from PubMed citations (Kilicoglu et al. 2012).
5. Monarch Initiative: A platform for disease-gene association data (Mungall et al. 2017).
6. ROBOKOP: A knowledge graph-based system for biomedical data integration and analysis (Bizon et al. 2019).

## Citation

If you use this dataset in your research, please cite the following paper:
```
@article{soman2023biomedical,
  title={Biomedical knowledge graph-enhanced prompt generation for large language models},
  author={Soman, Karthik and Rose, Peter W and Morris, John H and Akbas, Rabia E and Smith, Brett and Peetoom, Braian and Villouta-Reyes, Catalina and Cerono, Gabriel and Shi, Yongmei and Rizk-Jackson, Angela and others},
  journal={arXiv preprint arXiv:2311.17330},
  year={2023}
}
```
