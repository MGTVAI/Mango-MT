English | [简体中文](README_CN.md)

# Mango-MT: A 9B Model Bridging the Gap with Closed-Source Audiovisual MT

<img width="200" height="40" alt="5f135ff3e580ea95d407b9bd64df4d3d" src="https://github.com/user-attachments/assets/2adf1dc4-d567-4cbc-810f-d984bc57ee1f" />
<img width="200" height="70" alt="c1623a4589e3aeda320771b28fad45e6" src="https://github.com/user-attachments/assets/98eb2d27-b5f0-4e10-9a2f-3282123088e5" />


<img width="6303" height="1881" alt="f0" src="https://github.com/user-attachments/assets/68f72a09-0d9b-4e95-b9e6-f2a5a4d420e4" />



# Introduction

Global long-form video expansion requires robust multilingual subtitle translation, but generic MT fails at fragmented lines, timeline constraints and plot context for mass production. We open-sourced Mango-MT with 9B parameter size, an 11-language audiovisual subtitle translator paired with the benchmark called Mango-SubBench. Evaluations across all languages show it outperforms GPT, Gemini and DeepSeek with steady timeline compliance, consistent semantics and industrial scalability, see the [technical report](Technical_Report.pdf) for details. Our system holds four key advantages over prior work:

- **Scenario-oriented & Multilingual Optimization**: Professionally optimized for 11 different languages, perfectly adapting to complex industrial rules of video subtitle translation.
- **Context-aware Translation Mechanism**: Leverages global context modeling to stabilize plot logic and consistent character appellation translation.
- **Structural Robustness**: Rigidly retains original subtitle numbers and timestamps to ensure stable batch translation delivery.
- **Industrial-grade evaluation benchmark** : Unlike single-metric schemes, our benchmark integrates a full framework and high-quality dataset, evaluating subtitles against real delivery standards for reliable all-round quality assessment.


## News 🚀🚀🚀


- **2026/06/26** : 🚀 We introduce Mango-MT, an advanced machine learning large language model (MT) that demonstrates superior overall translation performance on film and television translation for 11 languages. Mango-MT achieves SoTA performance on Mango-SubBench, and our model matches or outperforms commercial models across most languages on Flores+. We released the model on  🤗 [Hugging Face](https://huggingface.co/MGTV-AI/Mango-MT-9B) , 🤖 [ModelScope](https://www.modelscope.cn/models/MGTVAI/Mango-MT-9B)
  
- **2026/06/22** : 🔥 We open-source the benchamark (Mango-SubBench) constructed on professional film and television translation data. We released the dataset on 🤗 [Hugging Face](https://huggingface.co/datasets/MGTV-AI/Mango-SubBench) , 🤖 [ModelScope](https://www.modelscope.cn/datasets/MGTVAI/Mango-SubBench)




## Installation
```
conda create -n mlt python==3.11
conda activate mlt
pip install -r requirements.txt
```

## Model Weights
Model checkpoints are accessible from [Hugging Face](https://huggingface.co/MGTV-AI/Mango-MT-9B) or [Modelscope](https://www.modelscope.cn/models/MGTVAI/Mango-MT-9B)

## Usage

### SGLang Server Usage
[SGLang](https://github.com/sgl-project/sglang) is a fast serving framework for large language models and vision language models. Please use the following command in a fresh environment:
```
uv pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python&egg=sglang[all]'
```
See its [documentation](https://docs.sglang.io/docs/get-started/install) for more details.

The following will create API endpoints at http://localhost:8000/v1:

```
python -m sglang.launch_server --model-path model_dir --port 8000 --tp-size 1 --mem-fraction-static 0.8 --context-length 4096 --reasoning-parser qwen3
```

### vLLM Server Usage
[vLLM](https://github.com/vllm-project/vllm) is a high-throughput and memory-efficient inference and serving engine for LLMs.  Please use the following command in a fresh environment:
```
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```
See its [documentation]([https://docs.sglang.io/docs/get-started/install](https://docs.vllm.ai/en/stable/getting_started/installation/index.html)) for more details.

For detailed usage guide, see the [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html).

The following will create API endpoints at http://localhost:8000/v1:
```
vllm serve model_dir --port 8000 --tensor-parallel-size 1 --max-model-len 4096 --reasoning-parser qwen3 --language-model-only
```

## Benchmark

### Mango-SubBench

Audiovisual subtitle translation evaluation prioritizes practical delivery standards for global long-form videos rather than mere similarity to human references. Qualified subtitles require simultaneous compliance with semantic accuracy, natural expression, standardized segmentation, valid timestamps and traceable quality, which cannot be fully assessed by single automatic metrics. To mitigate this limitation, we collaborate with Beijing International Studies University to build a dedicated audiovisual translation benchmark equipped with high-quality datasets and an automated-centric evaluation framework.It supports scalable, reproducible comprehensive model evaluation, with supplementary random manual sampling to assess subtle subtitle characteristics including character tone, cultural adaptability and viewing experience. In addition, a professional-grade dataset and evaluation benchmark tailored for audiovisual translation will be open-sourced in the near future. 

#### dataset

The evaluation set is constructed based on real-world film and television subtitle scenarios, covering 11 languages with a total of 8019 sentence-level samples.

| Language | Sample Count |
| --- | ---: |
| Indonesian(印尼语) | 1077 |
| English(英语) | 762 |
| Vietnamese(越南语) | 657 |
| Malay(马来语) | 489 |
| Thai(泰语) | 457 |
| Korean(韩语) | 699 |
| Japanese(日语) | 665 |
| Arabic(阿语) | 388 |
| French(法语)  | 1026|
| Spanish(西语)  | 523|
| Russian(俄语)  | 1026|
| All | 8019 |

#### Criteria

To enable horizontal comparison of different models and translation batches on the same evaluation set, we derive a weighted composite score with following metrics:

1. **Semantic similarity：**
   Semantic similarity quantifies the semantic proximity between model translations and reference translations based on text embeddings.
   
3. **COMET：**
   As a neural evaluation model, COMET integrates source text, human reference translation and machine translation output to predict the consistency between translations and human quality judgments. It is suitable for measuring whether the semantics of the source text are accurately preserved in translations.
   
5. **BLEU-2：**
   BLEU-2 provides supplementary information on local surface matching and phrase consistency. 


### Steps
#### Step 1
Download [eval data](https://huggingface.co/datasets/xxx) and put in /mt/benchmark/data/test_corpus.xlsx


#### Step 2
Translate with Mango-MT model by using following .py :

```
python /mt/benchmark/translate_with_mango_mt.py --input /mt/benchmark/data/test_corpus.xlsx --output /mt/benchmark/mango.xlsx
```

#### Step 3
Translate with DeepSeek-v4-pro, Gemini-3-pro, GPT-5.4 by using following .py :


```
python /mt/benchmark/translate_with_api.py --input /mt/benchmark/data/mango.xlsx --output /mt/benchmark/mango_deepseek_gemini_gpt.xlsx
```

#### Step 4
Evaluate the translated data to get the final evaluation score by using:

```
python /mt/benchmark/evaluate.py --input /mt/benchmark/mango_deepseek_gemini_gpt.xlsx
```

### Performance 
We evaluate our Mango-MT against three commercial large models (DeepSeek-v4-pro, Gemini-3-pro, GPT-5.4) across 11 languages with five core translation metrics: Semantic similarity, BLEU-2, COMET
On translation benchmark, Mango-MT outperforms Gemini, DeepSeek and GPT across all 11 evaluated languages. This demonstrates that our model delivers powerful multilingual translation capabilities for film and drama content, and possesses significant commercial deployment value. 

Evaluation Results on S(overall score) are: 

| Models          | Malay | Thai | English | Indonesian | Vietnamese | Russian | French | Japanese | Korean | Spanish | Arabic |
|-----------------|-------|------|---------|------------|------------|---------|--------|----------|--------|---------|--------|
| DeepSeek-V4-Pro | 0.76  | 0.80 | 0.85    | 0.75       | 0.85       | 0.76    | 0.79   | 0.80     | 0.70   | 0.82    | 0.78   |
| Gemini-3-Pro    | 0.78  | 0.88 | 0.86    | 0.79       | 0.87       | 0.79    | 0.82   | 0.82     | 0.73   | 0.83    | 0.78   |
| GPT-5.4         | 0.75  | 0.83 | 0.83    | 0.76       | 0.84       | 0.77    | 0.79   | 0.81     | 0.71   | 0.81    | 0.73   |
| Mango-MT        | **0.86**  | **0.91** | **0.90**| **0.93**  | **0.94** | **0.89**    | **0.90**   | **0.87**     | **0.81**   | **0.92**    | **0.81**   |


### FLORES+
We evaluate the multilingual translation performance of our model on FLORES+ which is based on FLORES-200. This dataset was originally released by FAIR researchers at Meta under the name FLORES.  The data is now being managed by OLDI, [the Open Language Data Initiative](https://oldi.org/). The + has been added to the name to disambiguate between the original datasets and this new actively developed version. For newer versions of this dataset, Please see [FLORES+ HuggingFace repo ](https://huggingface.co/datasets/openlanguagedata/flores_plus).
The data consists of translations primarily from English into over 200 language varieties. The original English sentences were sampled in equal amounts from [Wikinews](https://en.wikinews.org/wiki/Main_Page) (an international news source), [Wikijunior](https://en.wikibooks.org/wiki/Wikijunior) (a collection of age-appropriate non-fiction books), and [Wikivoyage](https://en.wikivoyage.org/wiki/Main_Page) (a travel guide).



### Performance 
We evaluate our Mango-MT against three commercial large models (DeepSeek-v4-pro, Gemini-3-pro, GPT-5.4) across 11 languages with two core translation metrics: BLEU and COMET.


| Models          | Metric | Russian | Indonesian | Japanese | French | Thai   | English | Spanish | Vietnamese | Arabic | Korean | Malay  | Avg    |
|-----------------|--------|---------|------------|----------|--------|--------|---------|---------|------------|--------|--------|--------|--------|
| DeepSeek-V4-Pro | BLEU   | 21.110  | 27.800     | 29.680   | 31.150 | 10.400 | 33.760  | 21.100  | 32.550     | 16.120 | 24.910 | 23.430 | 24.730 |
| DeepSeek-V4-Pro | COMET  | 0.945   | 0.940      | 0.925    | 0.921  | 0.894  | 0.977   | 0.945   | 0.921      | 0.895  | 0.914  | 0.908  | 0.926  |
| Gemini-3-Pro    | BLEU   | 22.840  | 28.850     | 31.640   | 32.840 | 9.480  | 34.090  | 22.410  | 32.730     | 17.300 | 25.220 | 24.090 | 25.590 |
| Gemini-3-Pro    | COMET  | 0.948   | 0.943      | 0.930    | 0.924  | 0.903  | 0.977   | 0.948   | 0.930      | 0.903  | 0.919  | 0.914  | 0.931  |
| GPT-5.4         | BLEU   | 20.260  | 24.980     | 27.710   | 28.070 | 8.380  | 30.990  | 21.390  | 30.080     | 13.240 | 23.110 | 21.120 | 22.670 |
| GPT-5.4         | COMET  | 0.946   | 0.945      | 0.928    | 0.925  | 0.904  | 0.978   | 0.950   | 0.928      | 0.895  | 0.920  | 0.919  | 0.931  |
| Mango-MT        | BLEU   | 20.370  | 22.060     | 27.350   | 32.980 | 10.620 | 30.110  | 21.640  | 30.060     | 10.450 | 20.440 | 20.040 | 22.370 |
| Mango-MT        | COMET  | 0.944   | 0.932      | 0.923    | 0.932  | 0.905  | 0.978   | 0.946   | 0.925      | 0.870  | 0.908  | 0.911  | 0.925  |





# Contact Us
If you love open source and enjoy tinkering, whether for learning purposes or to share better ideas, you are welcome to join us, contact with liuxusheng@mgtv.com

# About Joint Lab for Intelligent Media Translation
Co-founded by Mango TV and Beijing International Studies University, Joint Lab for Intelligent Media Translation aims to build a domestically leading R&D base for intelligent audiovisual translation technologies. Integrating technology research, talent cultivation and achievement commercialization as its core missions, the laboratory focuses on tackling key technical challenges including large-scale audiovisual translation models, dedicated evaluation benchmarks for such models, the construction of audiovisual translation corpora, and cultural compliance localization of translated media content.

# License
This framework is licensed under the Apache License. For models and datasets, please refer to the original resource page and follow the corresponding License.
