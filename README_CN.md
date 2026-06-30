[English](README.md) | 简体中文
# Mango-MT: A 9B Model Bridging the Gap with Closed-Source Audiovisual MT



<img width="6303" height="1881" alt="f0" src="https://github.com/user-attachments/assets/68f72a09-0d9b-4e95-b9e6-f2a5a4d420e4" />



# 介绍

面向全球的长视频出海业务，亟需一套高性能多语种字幕翻译方案；但通用机器翻译难以适配工业化量产场景下的碎片化台词、时间轴约束与剧情上下文理解难题。本文提出 Mango-MT—— 一款支持 11 种语言的视听字幕专用翻译模型，并配套构建 FT-MT 评测基准。全语种评测结果表明，该模型在时间轴合规稳定性、语义一致性与工程落地扩展性上全面优于 GPT、Gemini、DeepSeek 系列模型, 具体查看[技术报告](Technical_Report.pdf)。相较现有相关工作，我们的系统具备四大核心优势：

- **面向场景的多语种专项优化**: 针对 11 门语种完成专业化定制优化，完美适配视频字幕翻译复杂工业化规范。
- **上下文感知翻译机制**: 依托全局剧情上下文建模，保障剧情逻辑连贯、人物称谓翻译统一。
- **结构高鲁棒性**: 严格保留原始字幕编号与时间戳，实现批量翻译交付稳定可控。
- **工业级评测基准** : 区别于单一指标评测方案，本基准配套完整评测框架与高质量数据集，对标真实交付标准开展字幕评测，实现全面、可信的质量量化评估


## 新闻 🚀🚀🚀


- **2026/06/26** : 🚀 本文提出 Mango-MT，一款面向影视翻译场景、支持 11 门语种的先进大语言翻译模型，综合翻译性能表现优异。Mango-MT 在自研 FT-MT 评测基准上取得当前最优（SoTA）结果；在 Flores + 通用基准中，该模型在绝大多数语种上的效果持平乃至超越各类商用大模型。
  
- **2026/06/22** : 🔥 我们开源了基于专业影视翻译数据构建的评测基准 FT-MT。




## 安装
```
conda create -n mlt python==3.11
conda activate mlt
pip install -r requirements.txt
```

## 模型权重
从 xxx下载

## 使用

### SGLang 服务
[SGLang](https://github.com/sgl-project/sglang) 这是一套适用于大语言模型与多模态视觉语言模型的高速推理部署框架。请在全新的环境中执行以下命令：
```
uv pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python&egg=sglang[all]'
```
更多细节查看 [documentation](https://docs.sglang.io/docs/get-started/install)

执行下述操作后，将生成对应的 API 接口地址 http://localhost:8000/v1:

```
python -m sglang.launch_server --model-path model_dir --port 8000 --tp-size 1 --mem-fraction-static 0.8 --context-length 4096 --reasoning-parser qwen3
```

### vLLM 服务
[vLLM](https://github.com/vllm-project/vllm) 这是一款面向大语言模型（LLM）、具备高吞吐与显存高效利用特性的推理部署引擎。请在全新纯净环境中运行下述命令：
```
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```
更多细节查看 [documentation]([https://docs.sglang.io/docs/get-started/install](https://docs.vllm.ai/en/stable/getting_started/installation/index.html)) 

更多文档细节查看  [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html).

服务节点 http://localhost:8000/v1:
```
vllm serve model_dir --port 8000 --tensor-parallel-size 1 --max-model-len 4096 --reasoning-parser qwen3 --language-model-only
```

## 基准

### FT-MT

视听字幕翻译评测更看重长视频全球化出海场景下的实际交付标准，而非单纯追求译文与人工参考译文的字面相似度。合格字幕需要同时满足语义准确、表达自然、断句规范、时间戳有效、质量可追溯多项要求，仅依靠单一自动化指标无法完成完整评估。
为解决这一局限，我们联合北京第二外国语学院搭建了专用视听翻译评测数据集，配套高质量语料集与一套以自动化评测为核心的完整评测框架。该基准可实现可扩展、可复现的模型综合测评，同时辅以随机人工抽检，用于甄别字幕中人物语气、文化适配度、观影体验等细微文本特征。此外，一个适用于影视专业级的数据集和评估标准将会在未来一段时间开源出来。我们已在hugging face发布 [基准测试集](https://huggingface.co/datasets/xxx) 

#### 数据集

评测集基于真实影视字幕业务场景构建，覆盖 11 种语言，共计 8019 条句子级样本。
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

#### 标准

为实现不同模型、不同翻译批次在同一评测集下的横向对比，我们基于以下指标计算加权综合得分：

1. **语义相似度：**
   语义相似度通过文本嵌入向量，量化模型译文与参考译文之间的语义贴近程度。
   
3. **COMET：**
   COMET 是一款神经网络评测模型，它融合原文、人工参考译文与机器翻译输出，预测译文和人工质量评判标准的契合度，适用于衡量译文是否完整、准确保留原文语义。
   
5. **BLEU-2：**
   BLEU-2 可补充反映文本表层局部匹配度与短语一致性


### 步骤
#### 步骤 1
下载 [eval data](https://huggingface.co/datasets/xxx) 并放在 /mt/benchmark/data/test_corpus.xlsx


#### 步骤 2
使用下述 Python 脚本调用 Mango-MT 模型完成翻译：

```
python /mt/benchmark/translate_with_mango_mt.py --input /mt/benchmark/data/test_corpus.xlsx --output /mt/benchmark/result_1.xlsx
```

#### 步骤 3
利用下述 Python 脚本可完成基于DeepSeek-v4-pro, Gemini-3-pro, GPT-5.4三种模型翻译 :


```
python /mt/benchmark/translate_with_api.py --input /mt/benchmark/data/result_1.xlsx --output /mt/benchmark/result_2.xlsx
```

#### 步骤 4
运行以下代码对翻译结果进行评测，输出最终综合得分：

```
python /mt/benchmark/evaluate.py --input /mt/benchmark/result_2.xlsx
```

### 效果 
我们采用语义相似度、BLEU-2、COMET 五项核心翻译指标，在 11 种语言上对比评测 Mango-MT 与三款商用大模型（DeepSeek-v4-pro、Gemini-3-pro、GPT-5.4）。
在本次翻译评测基准测试中，Mango-MT 于全部 11 门被测语言上的表现均优于 Gemini、DeepSeek 与 GPT。这证明本模型针对影视剧集内容具备强劲的多语种翻译能力，拥有极高的商用落地价值。

综合得分 S 的评测结果如下：

| Models          | Malay | Thai | English | Indonesian | Vietnamese | Russian | French | Japanese | Korean | Spanish | Arabic |
|-----------------|-------|------|---------|------------|------------|---------|--------|----------|--------|---------|--------|
| DeepSeek-V4-Pro | 0.76  | 0.80 | 0.85    | 0.75       | 0.85       | 0.76    | 0.79   | 0.80     | 0.70   | 0.82    | 0.78   |
| Gemini-3-Pro    | 0.78  | 0.88 | 0.86    | 0.79       | 0.87       | 0.79    | 0.82   | 0.82     | 0.73   | 0.83    | 0.78   |
| GPT-5.4         | 0.75  | 0.83 | 0.83    | 0.76       | 0.84       | 0.77    | 0.79   | 0.81     | 0.71   | 0.81    | 0.73   |
| Mango-MT        | **0.86**  | **0.91** | **0.90**| **0.93**  | **0.94** | **0.89**    | **0.90**   | **0.87**     | **0.81**   | **0.92**    | **0.81**   |


### FLORES+

我们在基于 FLORES-200 拓展而来的 FLORES + 评测基准上测试本模型的多语言翻译性能。该数据集最初由 Meta 旗下 FAIR 实验室的研究人员发布，初代版本命名为 FLORES，如今由[开放语言数据计划](https://oldi.org/) OLDI负责维护管理。名称中增加 “+” 标识，是为了区分初代数据集与当前持续迭代更新的新版本。如需获取该数据集的最新版本，可访问 Hugging Face 平台的 FLORES + [数据集仓库](https://huggingface.co/datasets/openlanguagedata/flores_plus)。
该数据集以英文原文为基础，提供覆盖 200 余种语言变体的翻译文本。其中英文原文样本等量抽取自三类维基平台：国际新闻资讯网站[维基新闻](https://en.wikinews.org/wiki/Main_Page) 、面向青少年的科普读物维基少年[Wikijunior](https://en.wikibooks.org/wiki/Wikijunior) [旅行指南维基导游](https://en.wikivoyage.org/wiki/Main_Page)。

### 依赖

```
pip install sacrebleu sentencepiece
```
### 步骤
#### 步骤 1
下载devtest数据集  [flores devtest](https://huggingface.co/datasets/openlanguagedata/flores_plus) 并放在 reference_data/目录下. 

#### 步骤 2

将下载好的 cmn_Hans.jsonl 文件放入 src_data/ 目录，运行下方 Python 脚本调用自研模型，把中文翻译为 11 种语言；翻译后的结果会自动保存至 translated_data/ 目录。

```
cd /mt/flores200 && python translate_flores_with_mango_mt.py
```

同理，运行以下 Python 脚本调用 GPT、DeepSeek、Gemini 模型，将中文翻译为 11 种语言，翻译结果将保存至 translated_data / 目录中。

```
cd /mt/flores200 && python translate_flores_with_api.py
```
#### 步骤 3
我们运行下述 Python 脚本，采用两项通用核心指标（BLEU、chrF++）量化翻译质量：

```
python evaluate.py
```

### 效果 
我们采用两项核心翻译指标 ——BLEU（衡量短句流畅度，是字幕场景核心指标）与 chrF++（字符级语义匹配），在 11 种语言维度下，将自研 Mango-MT 与三款商用大模型（DeepSeek-v4-pro、Gemini-3-pro、GPT-5.4）开展对比评测。

BLEU评测指标结果:
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

chrF++评测指标结果:
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


1. **BLEU（字幕专项评测指标）：**
   Mango-MT 取得最高宏平均得分 26.89，优于 Gemini（25.59）、DeepSeek（24.73）与 GPT（22.67）。在 11 门语言中有 8 门语种得分排名第一，在英语及东南亚语种（印尼语、泰语、越南语、马来语）上优势尤为突出，高度适配字幕翻译业务场景。

2. **chrF++（细粒度语义评测指标）：**
   Gemini 宏平均得分 49.33 位居首位，Mango-MT 以 48.23 紧随其后，其次为 DeepSeek（47.89）、GPT（47.21）。在多门语种上，本模型表现与商用大模型持平甚至更优。

# 联系我们
如果你热爱开源、乐于钻研，无论是出于学习目的，或是希望交流优化思路，都欢迎加入我们。

# 致谢
感谢北京第二外国语学院为本项目字幕翻译相关工作提供支持

# 开源协议
本框架采用 Apache 开源许可协议。模型与数据集请查阅对应原始资源页面，并遵守其配套许可协议。
