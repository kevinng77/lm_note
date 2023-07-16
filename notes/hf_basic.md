## Huggingface

huggingface 中包含了几乎所有常用的 NLP Transformers 架构模型。所有的模型都在 [huggingface.co](https://huggingface.co/) 上可以查看到模型的开源信息。

我们先了解 Huggingface 模型的基础使用方法（如何调包），而后再去了解 HF 背后的 NLP 模型的运转逻辑，实现方法等。

### 使用 Huggingface 模型

使用 Transformers 模型都可以分成以下步骤：

1. 用户输入语句，`Tokenizer` 将文字转化为数字（`input_ids`）。这一步会使用到 Huggingface 中的 `AutoTokenizer` 相关 Class。

2. `input_ids` 传入到 `Model` 中，转化成为 `Embedding` （`Hidden_states`）。这一步会使用到 Huggingface 中的 `AutoModel` 相关 Class。

3. 根据 `Embedding`，我们根据任务进行解码，得到文字信息。这一步会使用到 `AutoModelForXXX` 相关 Class。

Step 1 及 Step 2 实现方法：

```python
from transformers import AutoTokenizer, AutoModel
model_id = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

text = "how are you."               
encoded_input = tokenizer(text, return_tensors='pt')      # Step 1
output = model(**encoded_input)                           # Step 2
print("模型的输出 output 包括以下内容:")
for key in output:
    value = getattr(output, key)
    print(f"{key} 是一个 shape = {value.shape} 的 Tensor")
# output
# 模型的输出 output 包括以下内容:
# last_hidden_state 是一个 shape = torch.Size([1, 6, 768]) 的 Tensor
# pooler_output 是一个 shape = torch.Size([1, 768]) 的 Tensor
```

#### 知识点 1

上述代码中，我们使用的是 `bert-base-uncased` 模型，这是一个 BERT 模型。他的源代码可以在 `src/transformers/models/bert` 文件夹下面找到，[点击这里查看](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert)。文件夹下面包括了 `modeling_bert.py`, `tokenization_bert.py` 等文件。

+ `tokenization_bert.py` 用于定义 Tokenizer，在第一步中，我们使用的是该文件下定义的 `BertTokenizer` 这个类。
+ `modeling_bert.py` 主要用于定义 BERT 模型的架构，再步骤二中，我们使用的就是该文件下定义的 `BertModel` 这个类。

是的，AutoTokenier 可以将他理解为一个复杂的映射，它能够根据你提供的 `model_id`，找到这个模型应该使用的 Class。

#### 知识点 2

Step 2 的输出可以看作是 input 的 embedding 结果，也称为 hidden_state。

output 中包含了很多数据，如 `last_hidden_state` （shape `[batch_size, len_seq, model_size]`）, `pooler_output` （shape `[batch_size, 1, model_size]`） 等 `embedding` 信息。

### **关于第三步：**

在 step1 和 step2 中，我们了解了如何调包，实现讲文本转换为 embedding。

通常不同的 NLP 任务，会在模型的 embedding 输出后，再加上一些 `LinearLayer` 以完成任务。比如我们要做情感分析任务，就需要添加一个 `linearlayer`，把 `embedding` 变成 3 维的，最后做一个 softmax 得到输出。

huggingface 提供好了这些 pipeline，分别包装在 `AutoModelForXXX` 类当中（[link 1](https://huggingface.co/docs/transformers/model_doc/auto)， [link 2](https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py#L1075)）：

比如对于情感分析，可以用 `AutoModelForSequenceClassification`：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
model_id = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_id)

config = AutoConfig.from_pretrained(model_id, num_labels=2  )      # 配置任务的参数
model = AutoModelForSequenceClassification.from_config(config)     # 加载任务模型

text = "How are you？"                                             # How many rows are there in the table?
encoded_input = tokenizer(text, return_tensors='pt')

output = model(**encoded_input)
output
```

LLM 通常用于对话任务，可以用：[AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) （适用于 `decoder` 模型）

```python
from transformers import AutoModelForCausalLM

model_id = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

text = "What to do on weekend?"
encoded_input = tokenizer(text, return_tensors='pt')
output = model.generate(**encoded_input, max_new_tokens=50)

print(tokenizer.decode(output[0]))
# What to do on weekend?

# The best way to do this is to do a quick test run of your code. This is a great way to get a feel for how your code is doing.

# The test run will take about 30 minutes.

# The test
```

模型输出如下：

```python
What to do on weekend?

The best way to do this is to do a quick test run of your code. This is a great way to get a feel for how your code is doing.

The test run will take about 30 minutes.

The test
```

也可以用 [AutoModelForSeq2SeqLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM) （适用于 `encoder-decoder` 模型）

```python
from transformers import AutoModelForSeq2SeqLM

model_id = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

text = "Write a plan for weekend."
encoded_input = tokenizer(text, return_tensors='pt')
output = model.generate(**encoded_input, max_new_tokens=50)
print(tokenizer.decode(output[0]))

# 输出
# <pad> During the weekend, you will be able to get a good night's sleep. If you are going to sleep, you will need to get a good night's sleep. If you are going to sleep, you will need
```

### 快速做一个 NLP Demo

1. **搜索：** 在 [huggingface.co task](https://huggingface.co/tasks) 上搜索你要做的任务，比如我们想要用于闲聊的模型。可以搜 NLP 下的 Conversational 任务。
2. **确认：** 在 models 页面查看模型信息，可以选下载量多的，一般我们需要查看模型大小、模型对应的类别、在什么数据上 finetune 过。
   - 好的模型，一般都会有详细的 model card（README 文件中的内容），如 [DialoGPT](https://huggingface.co/microsoft/DialoGPT-small/tree/main) 等。没有 modle card 的差[例子](https://huggingface.co/kevinng77/alpaca-7b-lora)。训练数据一般只能在 model card 找到。
   - 模型大小可以看 `Files and versions` 里面的 quanzhong 文件大小；模型类型可以在 `config.json` 中查看；
3. **使用：** 可以参考你所选模型对应的 model card；如果 model card 没写的话，可以根据模型类型，来判断要用 `AutoModelForSeq2SeqLM` 还是 `AutoModelForCausalLM`，而后参考。

## 后续活动

我们会在之后分享 HF 模型使用的高级技能，包括：

1. 模型训练

2. 上传自己的模型

3. 模型推理部署

