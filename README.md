# Survey of Cultural Awareness in Language Models: Text and Beyond
## (Being Updated)

> **[Survey of Cultural Awareness in Language Models: Text and Beyond](https://openreview.net/forum?id=3gg6GHhuvi)**[ [Link]](https://openreview.net/forum?id=3gg6GHhuvi)

> *Siddhesh Pawar<sup>2*</sup>, Junyeong Park<sup>1*</sup>, Jiho Jin<sup>1</sup>, Arnav Arora<sup>2</sup>, Junho Myung<sup>1</sup>, Srishti Yadav<sup>2</sup>, Faiz Ghifari Haznitrama<sup>1</sup>, Inhwa Song<sup>1</sup>, Alice Oh<sup>1</sup>, Isabelle Augenstein<sup>2</sup>*

> *<sup>1</sup>KAIST, <sup>2</sup>University of Copenhagen*


# 📁 Table of Content
- [Language Models and Culture](#language-models-and-culture)
- [Vision Models and Culture](#vision-models-and-culture)
- [Other Modalities and Culture](#other-modalities-and-culture)


# 💬 Language Models and Culture
## Cultural Alignment: Methodologies and Goals
### Training-Based Methods
### Training-Free Methods
### Goal-Specific Alignment Strategies

## Benchmarks and Evaluation
### Academic Knowledge
- ArabicMMLU: Assessing Massive Multitask Language Understanding in Arabic, <ins>ACL Findings, 2024</ins> [[Paper](https://aclanthology.org/2024.findings-acl.334/)] [[Code](https://github.com/mbzuai-nlp/ArabicMMLU)]
- CMMLU: Measuring massive multitask language understanding in Chinese, <ins>ACL Findings, 2024</ins> [[Paper](https://aclanthology.org/2024.findings-acl.671/)] [[Code](https://github.com/haonan-li/CMMLU)]
- Large language models only pass primary school exams in Indonesia: A comprehensive test on IndoMMLU, <ins>EMNLP, 2023</ins> [[Paper](https://aclanthology.org/2023.emnlp-main.760/)] [[Code](https://github.com/fajri91/IndoMMLU)]
- Should We Respect LLMs? A Cross-Lingual Study on the Influence of Prompt Politeness on LLM Performance, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.14531)] [[Code](https://github.com/nlp-waseda/JMMLU)]
- KMMLU: Measuring Massive Multitask Language Understanding in Korean, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.11548)] [[Code](https://huggingface.co/datasets/HAERAE-HUB/KMMLU)]
- TurkishMMLU: Measuring Massive Multitask Language Understanding in Turkish, <ins>EMNLP Findings, 2024</ins> [[Paper](https://arxiv.org/abs/2407.12402)] [[Code](https://github.com/ArdaYueksel/TurkishMMLU)]
- Khayyam Challenge (PersianMMLU): Is Your LLM Truly Wise to The Persian Language?, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.06644)] [[Code](https://github.com/raia-center/khayyam-challenge)]
- Gpt-4 can pass the korean national licensing examination for korean medicine doctors, <ins>PLOS Digital Health, 2023</ins> [[Paper](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000416)]
- CMB: A comprehensive medical benchmark in Chinese, <ins>NAACL, 2024</ins> [[Paper](https://aclanthology.org/2024.naacl-long.343/)] [[Code](https://github.com/FreedomIntelligence/CMB)]
- KorNAT: LLM Alignment Benchmark for Korean Social Values and Common Knowledge, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.13605)] [[Code](https://huggingface.co/datasets/jiyounglee0523/KorNAT)]
- Disce aut Deficere: Evaluating LLMs Proficiency on the INVALSI Italian Benchmark, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.17535)] [[Code](https://huggingface.co/spaces/Crisp-Unimib/INVALSIbenchmark)]
- FoundaBench: Evaluating Chinese Fundamental Knowledge Capabilities of Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.18359)]
- M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models, <ins>NeurIPS (Datasets and Benchmarks), 2023</ins> [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/117c5c8622b0d539f74f6d1fb082a2e9-Abstract-Datasets_and_Benchmarks.html)] [[Code](https://github.com/DAMO-NLP-SG/M3Exam)]


### Commonsense Knowledge
#### Culture-Specific
- CIF-Bench: A Chinese Instruction-Following Benchmark for Evaluating the Generalizability of Large Language Models, <ins>ACL, 2024</ins> [[Paper](https://arxiv.org/abs/2402.13109)] [[Code](https://github.com/yizhilll/CIF-Bench)]
- FoundaBench: Evaluating Chinese Fundamental Knowledge Capabilities of Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.18359)]
- COPAL-ID: Indonesian Language Reasoning with Local Culture and Nuances, <ins>NAACL, 2024</ins> [[Paper](https://aclanthology.org/2024.naacl-long.77/)] [[Code](https://github.com/haryoa/COPAL-ID)]
- IndoCulture: Exploring Geographically-Influenced Cultural Commonsense Reasoning Across Eleven Indonesian Provinces, <ins>TACL, 2024</ins> [[Paper](https://arxiv.org/abs/2404.01854)] [[Code](https://huggingface.co/datasets/indolem/IndoCulture)]
- BHASA: A Holistic Southeast Asian Linguistic and Cultural Evaluation Suite for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.06085)] [[Code](https://github.com/aisingapore/BHASA)]
- Can LLM Generate Culturally Relevant Commonsense QA Data? Case Study in Indonesian and Sundanese, <ins>EMNLP, 2024</ins> [[Paper](https://arxiv.org/abs/2402.17302)] [[Code](https://github.com/rifkiaputri/id-csqa)]
- CLIcK: A Benchmark Dataset of Cultural and Linguistic Intelligence in Korean, <ins>LREC | COLING, 2024</ins> [[Paper](https://aclanthology.org/2024.lrec-main.296/)] [[Code](https://github.com/rladmstn1714/CLIcK)]
- HAE-RAE Bench: Evaluation of Korean Knowledge in Language Models, <ins>LREC | COLING, 2024</ins> [[Paper](https://aclanthology.org/2024.lrec-main.704/)] [[Code](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH_1.1)]
- AraDiCE: Benchmarks for Dialectal and Cultural Capabilities in LLMs, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2409.11404)]
- BertaQA: How Much Do Language Models Know About Local Culture?, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.07302)] [[Code](https://github.com/juletx/BertaQA)]
- DOSA: A Dataset of Social Artifacts from Different Indian Geographical Subcultures, <ins>LREC | COLING, 2024</ins> [[Paper](https://aclanthology.org/2024.lrec-main.474/)] [[Code](https://github.com/microsoft/DOSA)]
- "Vorbeşti Româneşte?" A Recipe to Train Powerful Romanian LLMs with English Instructions, <ins>EMNLP Findings, 2024</ins> [[Paper](https://arxiv.org/abs/2406.18266)] [[Code](https://openllm.ro/)]
- Benchmarking Cognitive Domains for LLMs: Insights from Taiwanese Hakka Culture, <ins>O-COCOSDA, 2024</ins> [[Paper](https://arxiv.org/abs/2409.01556)]
- PARIKSHA: A Large-Scale Investigation of Human-LLM Evaluator Agreement on Multilingual and Multi-Cultural Data, <ins>EMNLP, 2024</ins> [[Paper](https://arxiv.org/abs/2406.15053)] [[Code](https://github.com/WattsIshaan/PARIKSHA)]

#### Multiculture Monolingual
- Extracting Cultural Commonsense Knowledge at Scale, <ins>WWW, 2023</ins> [[Paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583535)] [[Code](https://candle.mpi-inf.mpg.de/)]
- Having Beer after Prayer? Measuring Cultural Bias in Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.14456)] [[Code](https://github.com/tareknaous/camel)]
- Massively Multi-Cultural Knowledge Acquisition & LM Benchmarking, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.09369)] [[Code](https://github.com/yrf1/LLM-MassiveMulticultureNormsKnowledge-NCLB)]
- CulturalTeaming: AI-Assisted Interactive Red-Teaming for Challenging LLMs' (Lack of) Multicultural Knowledge, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.06664)] [[Code](https://huggingface.co/datasets/kellycyy/CulturalBench)]
- CULTURE-GEN: Revealing Global Cultural Perception in Language Models through Natural Language Prompting, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.10199)] [[Code](https://github.com/huihanlhh/Culture-Gen/)]
- How Well Do LLMs Identify Cultural Unity in Diversity?, <ins>COLM, 2024</ins> [[Paper](https://arxiv.org/abs/2408.05102)] [[Code](https://github.com/ljl0222/CUNIT)]
- CPopQA: Ranking Cultural Concept Popularity by LLMs, <ins>NAACL, 2024</ins> [[Paper](https://aclanthology.org/2024.naacl-short.52/)] [[Code](https://github.com/SeleenaJM/CPopQA)]
- DLAMA: A Framework for Curating Culturally Diverse Facts for Probing the Knowledge of Pretrained Language Models, <ins>ACL Findings, 2023</ins> [[Paper](https://aclanthology.org/2023.findings-acl.389/)] [[Code](https://github.com/AMR-KELEG/DLAMA)]
- EnCBP: A New Benchmark Dataset for Finer-Grained Cultural Background Prediction in English, <ins>ACL Findings, 2022</ins> [[Paper](https://aclanthology.org/2022.findings-acl.221/)]
- FORK: A Bite-Sized Test Set for Probing Culinary Cultural Biases in Commonsense Reasoning Models, <ins>ACL Findings, 2023</ins> [[Paper](https://aclanthology.org/2023.findings-acl.631/)] [[Code](https://github.com/shramay-palta/FORK_ACL2023)]
- Global-Liar: Factuality of LLMs over Time and Geographic Regions, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.17839)]

#### Multiculture Multilingual
- A diverse Multilingual News Headlines Dataset from around the World, <ins>NAACL, 2024</ins> [[Paper](https://aclanthology.org/2024.naacl-short.55/)] [[Code](https://huggingface.co/datasets/felixludos/babel-briefings)]
- BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages, <ins>NeurIPS (Datasets and Benchmarks), 2024</ins> [[Paper](https://arxiv.org/abs/2406.09948)] [[Code](https://github.com/nlee0212/BLEnD)]
- CaLMQA: Exploring culturally specific long-form question answering across 23 languages, <ins>CoRR, 2024</ins> [[Paper](https://arxiv.org/abs/2406.17761)] [[Code](https://github.com/2015aroras/CaLMQA)]
- Cultural Adaptation of Recipes, <ins>TACL, 2024</ins> [[Paper](https://aclanthology.org/2024.tacl-1.5/)] [[Code](https://github.com/coastalcph/cultural-recipes)]
- Does Mapo Tofu Contain Coffee? Probing LLMs for Food-related Cultural Knowledge, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.06833)] [[Code](https://github.com/lizhou21/FmLAMA-master)]
- GeoMLAMA: Geo-Diverse Commonsense Probing on Multilingual Pre-Trained Language Models, <ins>EMNLP, 2022</ins> [[Paper](https://aclanthology.org/2022.emnlp-main.132/)] [[Code](https://github.com/WadeYin9712/GeoMLAMA)]
- Are Multilingual LLMs Culturally-Diverse Reasoners? An Investigation into Multicultural Proverbs and Sayings, <ins>NAACL, 2024</ins> [[Paper]([paper_link](https://aclanthology.org/2024.naacl-long.112/))] [[Code]([code_link](https://github.com/UKPLab/maps))]
- NativQA: Multilingual Culturally-Aligned Natural Query for LLMs, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.09823)] [[Code](https://nativqa.gitlab.io/)]
- OMGEval: An Open Multilingual Generative Evaluation Benchmark for Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.13524)] [[Code](https://github.com/blcuicall/OMGEval)]
- SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning, <ins>NAACL, 2024</ins> [[Paper](https://aclanthology.org/2024.naacl-long.22/)] [[Code](https://github.com/SeaEval/SeaEval)]
- Large Language Models are Geographically Biased, <ins>ICML Poster, 2024</ins> [[Paper](https://arxiv.org/abs/2402.02680)] [[Code](https://rohinmanvi.github.io/GeoLLM/)]
- Understanding the Capabilities and Limitations of Large Language Models for Cultural Commonsense, <ins>NAACL, 2024</ins> [[Paper](https://aclanthology.org/2024.naacl-long.316/)]
- Good Night at 4 pm?! Time Expressions in Different Cultures, <ins>ACL Findings, 2022</ins> [[Paper](https://aclanthology.org/2022.findings-acl.224/)] [[Code](https://github.com/vered1986/time_expressions)]
- Not All Countries Celebrate Thanksgiving: On the Cultural Dominance in Large Language Models, <ins>CoRR, 2023</ins> [[Paper](https://arxiv.org/abs/2310.12481)]

### Social Values
- Value FULCRA: Mapping Large Language Models to the Multidimensional Spectrum of Basic Human Value, <ins>NAACL, 2024</ins> [[Paper](https://aclanthology.org/2024.naacl-long.486/)]
- The Ghost in the Machine has an American accent: value conflict in GPT-3, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2203.07785)]
- CDEval: A Benchmark for Measuring the Cultural Dimensions of Large Language Models, <ins>ACL C3NLP Workshop, 2024</ins> [[Paper](https://arxiv.org/abs/2311.16421)] [[Code](https://huggingface.co/datasets/Rykeryuhang/CDEval)]
- High-Dimension Human Value Representation in Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.07900)] [[Code](https://github.com/HLTCHKUST/UniVaR)]
- Probing Pre-Trained Language Models for Cross-Cultural Differences in Values, <ins>ACL C3NLP Workshop, 2023</ins> [[Paper](https://aclanthology.org/2023.c3nlp-1.12/)] [[Code](https://github.com/copenlu/value-probing)]
- Assessing Cross-Cultural Alignment between ChatGPT and Human Societies: An Empirical Study, <ins>ACL C3NLP Workshop, 2023</ins> [[Paper](https://aclanthology.org/2023.c3nlp-1.7/?trk=public_post_comment-text)] [[Code](https://github.com/yongcaoplus/ProbingChatGPT)]
- WorldValuesBench: A Large-Scale Benchmark Dataset for Multi-Cultural Value Awareness of Language Models, <ins>LREC | COLING, 2024</ins> [[Paper](https://aclanthology.org/2024.lrec-main.1539/)] [[Code](https://github.com/Demon702/WorldValuesBench)]
- Not All Countries Celebrate Thanksgiving: On the Cultural Dominance in Large Language Models, <ins>CoRR, 2023</ins> [[Paper](https://arxiv.org/abs/2310.12481)]
- Cultural Value Resonance in Folktales: A Transformer-Based Analysis with the World Value Corpus, <ins>SBP-BRiMS, 2022</ins> [[Paper](https://dl.acm.org/doi/abs/10.1007/978-3-031-17114-7_20)] [[Code](https://osf.io/wpu8r)]
- CIVICS: Building a Dataset for Examining Culturally-Informed Values in Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.13974)] [[Code](https://huggingface.co/CIVICS-dataset)]
- KorNAT: LLM Alignment Benchmark for Korean Social Values and Common Knowledge, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.13605)] [[Code](https://huggingface.co/datasets/jiyounglee0523/KorNAT)]
- LocalValueBench: A Collaboratively Built and Extensible Benchmark for Evaluating Localized Value Alignment and Ethical Safety in Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2408.01460)]
- Investigating Human Values in Online Communities, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.14177)] [[Code](https://github.com/copenlu/HumanValues)]
- Building Knowledge-Guided Lexica to Model Cultural Variation, <ins>NAACL, 2024</ins> [[Paper](https://aclanthology.org/2024.naacl-long.12/)]

### Social Norms and Morals
### Social Bias and Stereotype
### Toxicity and Safety
### Emotional and Subjective Topics
### Linguistics


# 🖼️ Vision Models and Culture


# 🎞️ Other Modalities and Culture
