# Mango-MT: A 9B Model Bridging the Gap with Closed-Source Audiovisual MT

<img width="6303" height="1881" alt="f0" src="https://github.com/user-attachments/assets/68f72a09-0d9b-4e95-b9e6-f2a5a4d420e4" />



# Introduction

Global long-form video expansion requires robust multilingual subtitle translation, but generic MT fails at fragmented lines, timeline constraints and plot context for mass production. We introduce Mango-MT, an 11-language audiovisual subtitle translator paired with the FT-MT benchmark. Evaluations across all languages show it outperforms GPT, Gemini and DeepSeek with steady timeline compliance, consistent semantics and industrial scalability. Our system holds four key advantages over prior work:

- **Scenario-oriented & Multilingual Optimization**: Professionally optimized for 11 different languages, perfectly adapting to complex industrial rules of video subtitle translation.
- **Context-aware Translation Mechanism**: Leverages global context modeling to stabilize plot logic and consistent character appellation translation.
- **Structural Robustness**: Rigidly retains original subtitle numbers and timestamps to ensure stable batch translation delivery.
- **Industrial-grade evaluation benchmark** : Unlike single-metric schemes, our benchmark integrates a full framework and high-quality dataset, evaluating subtitles against real delivery standards for reliable all-round quality assessment.


## News 🚀🚀🚀


- **2026/06/26** : 🚀 We introduce Mango-MT, an advanced machine learning large language model (MT) that demonstrates superior overall translation performance on film and television translation for 11 languages. Mango-MT achieves SoTA performance on FTT-MT, and our model matches or outperforms commercial models across most languages on Flores+.
  
- **2026/06/22** : 🔥 We open-source the benchamark (FTT-MT) constructed on professional film and television translation data.




## Installation
```
conda create -n mlt python==3.11
conda activate mlt
pip install -r requirements.txt
```

## Model Weights
Model checkpoints are accessible from xxx

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

### FT-MT

Audiovisual subtitle translation evaluation prioritizes practical delivery standards for global long-form videos rather than mere similarity to human references. Qualified subtitles require simultaneous compliance with semantic accuracy, natural expression, standardized segmentation, valid timestamps and traceable quality, which cannot be fully assessed by single automatic metrics. To mitigate this limitation, we collaborate with Beijing International Studies University to build a dedicated audiovisual translation benchmark equipped with high-quality datasets and an automated-centric evaluation framework.It supports scalable, reproducible comprehensive model evaluation, with supplementary random manual sampling to assess subtle subtitle characteristics including character tone, cultural adaptability and viewing experience. We have released the [benchmark datasets](https://huggingface.co/datasets/xxx) in hugging face.

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
   BLEU-2 provides supplementary information on local surface matching and phrase consistency. Thus, its weight is doubled in the weighted composite score.


### Steps
#### Step 1
Download [eval data](https://huggingface.co/datasets/xxx) and put in /mt/benchmark/data/test_corpus.xlsx


#### Step 2
Translate with Mango-MT model by using following .py :

```
python /mt/benchmark/translate_with_mango_mt.py --input /mt/benchmark/data/test_corpus.xlsx --output /mt/benchmark/result_1.xlsx
```

#### Step 3
Translate with DeepSeek-v4-pro, Gemini-3-pro, GPT-5.4 by using following .py :


```
python /mt/benchmark/translate_with_api.py --input /mt/benchmark/data/result_1.xlsx --output /mt/benchmark/result_2.xlsx
```

#### Step 4
Evaluate the translated data to get the final evaluation score by using:

```
python /mt/benchmark/evaluate.py --input /mt/benchmark/result_2.xlsx
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


### Dependencies

```
pip install sacrebleu sentencepiece
```
### Steps
#### Step 1
Download the devtest datasets of following languages from [flores devtest](https://huggingface.co/datasets/openlanguagedata/flores_plus) and put them into reference_data/. 

#### Step 2

Put the downloaded cmn_Hans.jsonl into the src_data/, translate Chinese into 11 languages by using following .py based on our model, the translated results will be saved into the translated_data/:

```
cd /mt/flores200 && python translate_flores_with_Mango_MT.py
```

Similarly, translate Chinese into 11 languages by using following .py based on gpt, deepseek, gemini, the translated results will be saved into the translated_data/:

```
cd /mt/flores200 && python translate_flores_with_api.py
```
#### Step 3
we adopt two core universal metrics (BLEU, chrF++ ) to quantify translation quality by using following .py:

```
python evaluate.py
```

### Performance 
We evaluate our Mango-MT against three commercial large models (DeepSeek-v4-pro, Gemini-3-pro, GPT-5.4) across 11 languages with two core translation metrics: BLEU (short utterance fluency, core metric for subtitles) and chrF++ (character-level semantic matching).

Evaluation Results on  BLEU:
|Language|DeepSeek|Gemini|GPT|Mango\-MT|
|---|---|---|---|---|
|Russian|21\.11|22\.84|20\.26|**23\.33**|
|Indonesian|27\.80|28\.85|24\.98|**31\.75**|
|Japanese|29\.68|**31\.64**|27\.71|30\.99|
|French|31\.15|32\.84|28\.07|**33\.83**|
|Thai|10\.40|9\.48|8\.38|**10\.52**|
|English|33\.76|34\.09|30\.99|**40\.49**|
|Spanish|21\.10|22\.41|21\.39|**24\.57**|
|Vietnamese|32\.55|32\.73|30\.08|**34\.23**|
|Arabic|**16\.12**|17\.30|13\.24|16\.10|
|Korean|24\.91|**25\.22**|23\.11|24\.48|
|Malay|23\.43|24\.09|21\.12|**25\.52**|
|**Macro Average**|24\.73|25\.59|22\.67|**26\.89**|

Evaluation Results on chrF++:
|Language|DeepSeek|Gemini|GPT|Mango\-MT|
|---|---|---|---|---|
|Russian|**48\.23**|49\.85|**48\.23**|48\.30|
|Indonesian|56\.65|**58\.65**|55\.82|57\.97|
|Japanese|**30\.40**|30\.08|29\.01|28\.64|
|French|56\.15|**58\.36**|55\.21|57\.33|
|Thai|43\.53|**45\.24**|43\.49|42\.83|
|English|60\.00|61\.00|58\.84|**63\.78**|
|Spanish|48\.07|49\.28|48\.48|**49\.60**|
|Vietnamese|53\.08|**54\.42**|52\.55|53\.55|
|Arabic|45\.37|**47\.26**|43\.59|43\.99|
|Korean|31\.12|**32\.52**|30\.15|30\.22|
|Malay|54\.19|**55\.99**|53\.95|54\.25|
|**Macro Average**|47\.89|**49\.33**|47\.21|48\.23|


1. **BLEU (subtitle-focused metric)：**
   Mango-MT achieves the highest macro average score (26.89), outperforming Gemini (25.59), DeepSeek (24.73) and GPT (22.67). It ranks first on 8 out of 11 languages, especially showing dominant advantages on English and Southeast Asian languages (Indonesian, Thai, Vietnamese, Malay), which fits our subtitle translation scenario perfectly.

2. **chrF++ (fine-grained semantic metric)：**
   Gemini leads the macro average (49.33), followed by Mango-MT (48.23), DeepSeek (47.89) and GPT (47.21). Our model matches or outperforms commercial models across multiple languages.

# Contact Us
If you love open source and enjoy tinkering, whether for learning purposes or to share better ideas, you are welcome to join us.

# Acknowledgements
Thanks to the support of subtitle translation from Beijing International Studies University 

# License
This framework is licensed under the Apache License. For models and datasets, please refer to the original resource page and follow the corresponding License.
