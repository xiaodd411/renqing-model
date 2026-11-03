# 底座说明

- 原始模型：HFL / `hfl/rbt3`
- 模型页面：https://huggingface.co/hfl/rbt3
- 本次下载版本：`0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c`
- 原作者：Joint Laboratory of HIT and iFLYTEK Research (HFL)
- 模型页面声明的许可：Apache-2.0
- 原始研究：[Revisiting Pre-Trained Models for Chinese Natural Language Processing](https://aclanthology.org/2020.findings-emnlp.58/)
- 原仓库：https://github.com/ymcui/Chinese-BERT-wwm

本项目在原始编码器上进行中文人际场景的监督微调，并新增候选打分头和独立温度校准。训练后的编码器并非未经修改的 HFL 原模型。HFL、TypeSafe AI 与本项目没有关联或背书关系。

## 可选指令模型底座

- 原始模型：[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- 固定版本：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- 原作者：Qwen Team
- 模型许可：Apache-2.0；运行 `scripts/fetch_causal_base.py` 时随固定版本下载原始 `LICENSE`。
- 底座权重 SHA256：`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`

新训练产物为本项目监督微调的 LoRA 适配器，底座保持冻结；完整项目包不重复分发约 3.1GB 的底座权重。Qwen 与本项目没有关联或背书关系。
