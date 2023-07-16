# LLM 训练数据制作手册(一)

## 回顾

我们讨论到了指令微调的数据集：

### Alpaca

[standford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) 对 LLaMa 进行了微调，并尝试使用 self-instruct 方法对 gpt-turbo-3.5 进行 **蒸馏学习**。

**数据制作方法：** Alpaca 修改了 [self-instruct](https://arxiv.org/abs/2212.10560) 提供的方案，并使用 OPENAI API 生成了 52k 的 instruction tuning 数据集，共花费 $500。所谓 Alpaca 的 self-instruct 方案，就是将下面这串prompt 发送给 gpt-turbo-3.5，而后从中抽取我们需要的训练结果。

```txt
You are asked to come up with a set of 10 diverse task instructions. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instructions.
... 一些其他限制
9. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.

List of 10 tasks:

###
1. Instruction: Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?
1. Inputs: <noinput>
1. Output: Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams whey protein powder, 1/2 medium banana, 1tbsp flaxseed oil and 1/2 cup water, totalling about 550 calories. The 4 strips of bacon contains about 200 calories.

###
2. Instruction: Generate an appropriate subjective title for the following email:
2. Inputs: Hi [person name],I'm writing to ask you if you are happy to be a panelist of our workshop on multimodality at CVPR. The workshop will be held on June 20, 2023. \n\nBest,\n[my name]
2. Output: Invitition to be a panelist for CVPR 2023 workshop on Multimodality

### 
3. Instruction: 
```

基于以上的 prompt，每次发送 OPENAI API 请求后，GPT-4 或者 gpt-turbo-3.5 会续写 10+ 个相关的新任务和输出。

我们可以限定 GPT 生成的任务类型，比如让他只生成中文数据，只生成 TEXT-TO-SQL 相关数据等等，实现方法就是在 prompt 当中和 GPT 说。

**效果：** 官方仓库中并没有很严谨的评判，仅让 5 位裁判员根据几个示例样本来对 Alpaca，llama 还有 text-davinci-003 进行打分。参考 [HF LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 还有 [chat Arena](https://chat.lmsys.org/?arena)，Alpaca 相对 LLaMa 的提升还是有限。

Alpaca 推出后，后许多根据 Alpaca 修改的高效微调方案，比如 Alpaca-lora，Alpaca-Adapter 等等。关于微调训练，我们会在下周和大家讨论。

### Vicuna

[vicuna](https://github.com/lm-sys/FastChat/tree/main) 的效果在 Open Source LLM 中数一数二，目前公开了 7B, 13B, 33B 三个版本。

**数据：** 数据来源于 shareGPT 的用户多轮对话数据，vicuna 对[原始对话数据集](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)进行了清洗，筛选了约 70k 数据进行训练。所谓 shareGPT 的用户多轮对话数据，就是用户上传的 GPT 对话记录。

**效果：** LMSYS 设计了让模型和模型之间 PK 的测评方式，并依据这种方式设计了类似排位赛的 [Chat arena](https://chat.lmsys.org/?arena) 。参考 LMSYS 发布的这个排行榜，Vicuna 的效果会比 Koala, MPT 等大多数模型好。

>  如何对多轮对话训练，是一个很关键的NLP 知识点，我们会在下文中和大家介绍！

### WizardLM

[wizardLM](https://github.com/nlpxucan/WizardLM) 的效果和 Vicuna 差不多，但 WizardLM 更侧重于 instruction 任务，vicuna 更侧重于对话任务。所谓的 instruction 任务，可以理解为命令性的任务，比如“为我生成10条请假理由”。

**数据：** WizardLM 通过 [evol-instruct](https://github.com/nlpxucan/evol-instruct) 方法，对 alpaca 的 self-instruct 数据集进行了优化（修改了使用 GPT4 生成样本时候的 prompt，让训练样本变得复杂），整个 [evol-instruct 数据](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)集共 196k。

所谓的 evol-instruct，其实就是升级了以下 alpaca 中生成数据的 prompt，大家可以看下面这个模板：

```text
I want you act as a Prompt Rewriter. Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle. 

But the rewritten prompt must be reasonable and must be understood and responded by humans. 

Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#. You SHOULD complicate the given prompt using the following method: 
If #Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased. 
or You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#. 
‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’ are not allowed to appear in #Rewritten Prompt# #Given Prompt#: 
<Here is instruction.> 
#Rewritten Prompt#:
```

**效果：** 官方指标对比了 GPT-4 Evaluation ， MMLU，ARC 等指标。同参数量级的 WizardLM 会和 Vicuna 效果差不多。

Wizard 仓库还开源了 WizardCoder 等模型，是基于 Startcoder 进行训练的，主打代码能力的模型。此外，笔者对几个开源的 Wizard 模型进行了 MMLU 测试（[TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ](TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ)，），分数都低地奇怪，只有 35+，不确定是什么原因。