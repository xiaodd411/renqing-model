# 人情世故决策模型 Renqing

输入中文人际场景和一组有明确答案空间的问题，输出候选概率、期望分数或真假概率。项目包含原创演示数据、真实神经网络训练、独立校准、测试评估和离线推理。

首版 240 条合成数据的历史实测结论见 [RESULTS.md](RESULTS.md)。扩展数据为 24,396 条，来源与计数口径见 [data/DATA_SOURCES.md](data/DATA_SOURCES.md)。新增公开回答立场标签和语境对照数据，仍属于需要独立人工验证的实验模型。

扩展训练实测见 [RESULTS_V2.md](RESULTS_V2.md)：公开立场任务改善，但非模板化人际挑战没有可信提升，部分任务退步。默认权重继续保留 v0.1，新权重仅供显式选择的实验使用。

参考 [TypeSafe 官方 Jev 介绍](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 和[问题类型文档](https://docs.typesafe.ai/introduction)，采用类似的类型化决策思路。这里的模型结构、训练方法和请求协议由本项目定义；训练使用软标签交叉熵与温度缩放，没有实现或声称复现 TypeSafe 的 RLCD、采样器及性能。

## 开始使用

在 PowerShell 中进入本项目目录：

```powershell
# 项目内独立安装，不安装系统级依赖。需要已有 uv。
.\setup.ps1
# 无 NVIDIA CUDA 环境时使用 .\setup.ps1 -Cpu

# 直接使用已训练权重（如果 runs/renqing-v1 已随项目提供）
.\predict.ps1

# 重新训练：新输出目录，避免覆盖已有结果
.\train.ps1 -Output runs\my-first-run -Epochs 20
```

Git 仓库保存源码、演示数据和首轮训练评估记录；约 154MB 的编码器权重通过 Release 的完整项目包提供。克隆仓库后，如需使用现成权重，请下载 Release 附件，解压后将其中 `runs/renqing-v1` 复制到仓库的同名目录。也可以直接解压完整项目包并运行。

安装时会从官方 PyTorch 索引下载依赖，并从 Hugging Face 下载公开的 `hfl/rbt3` 底座；训练和推理在本机执行，场景数据不提交给外部推理 API。已保存的模型可完全离线加载。

在本次生成的原始工作目录中，`predict.ps1` 和 `train.ps1` 也会自动使用已经装好的 `work/.venv`，可直接运行。复制项目到其他位置后先执行 `setup.ps1`。`requirements-tested.txt` 记录本次实测的依赖版本；CUDA 版 torch 的下载索引仍须按 `setup.ps1` 选择。

也可以手动调用 Python：

```powershell
.venv\Scripts\python.exe -m renqing.cli validate --data data\demo
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m renqing.cli predict --model runs\renqing-v1 --input examples\request.json
.venv\Scripts\python.exe -m renqing.cli evaluate --model runs\renqing-v1 --data data\demo\test.jsonl --output runs\independent-evaluation.json
```

`--output` 指定的推理/评估文件必须不存在；训练目标目录必须为空。模型路径和输入文件路径均相对当前工作目录。JSON 文件使用 UTF-8。

## 请求与结果

[examples/request.json](examples/request.json) 同时演示三个问题：选择更得体的回应、判断是否有足够证据推断恶意、评估当前冲突风险。可以直接修改其中的场景和候选项。

| 类型 | 问题定义 | 返回值 |
| --- | --- | --- |
| `choice` | `prompt`、2–16 个不重复的 `options` | `choice`、`index`、候选项顺序一致的 `probabilities` |
| `noul` | 一个原子的是非问题 `prompt` | `noul=P(是)`；`probabilities` 固定为 `[P(是), P(否)]` |
| `score` | `prompt`、有序等级 `options`、严格递增的数字 `values` | `score=ΣP(等级)×对应数值`、最高概率等级 `level` |

每种类型都附带 `confidence=max(probabilities)` 和 `needs_review`。默认复核阈值是 0.8；未校准的模型始终标记复核。这里的 `confidence` 是最大类别概率，不是 Jev 的独立认知不确定性估计。温度在单独的校准集上拟合，并不保证新领域上的概率可靠。

示例：

```python
from renqing.model import Predictor

predictor = Predictor("runs/renqing-v1", device="auto")
result = predictor.predict({
    "state": "朋友只回了一个好，没有其他上下文。",
    "questions": {
        "anger": {"type": "noul", "prompt": "仅凭这条回复，能确定朋友正在生气吗？"}
    }
})
print(result)
```

模型为每个 `(场景, 问题, 候选项)` 独立编码并输出一个实数，再在同一问题内 softmax。输入的问题互不影响；候选顺序改变只会改变概率的顺序。它对候选集合内的相对偏好建模，不能判断未提供的选项。添加或删去选项可能改变分布。

## 扩展数据 v0.2

| 集合 | 决策记录 | 隔离分组 |
| --- | ---: | ---: |
| train | 18,020 | 3,031 |
| validation | 2,124 | 286 |
| calibration | 2,104 | 290 |
| test | 2,148 | 439 |
| 合计 | 24,396 | 4,046 |

这些记录来自 4,945 个公开问答对、48 个合成语义场景族的 864 种文本、56 个旧训练场景和 32 个新冻结挑战场景。总行数不代表独立人际事件数量。公开任务测量回答的观点立场，人际标签仍需独立人工复核。

```powershell
# 直接使用已生成的数据训练，约 12GB NVIDIA 显存可运行本次配置
.\train-expanded.ps1 -Output runs\my-expanded-run

# 使用本次扩展数据训练得到的权重
.\predict.ps1 -Model runs\renqing-v2

# 从官方公开归档复现扩展数据；首次下载约 135MB
.venv\Scripts\python.exe scripts\build_challenge_v2.py
.venv\Scripts\python.exe scripts\prepare_expanded.py
```

数据构建脚本、审计脚本和挑战集构建脚本均拒绝覆盖已有输出；复现时请使用空目录或显式给新的 `--output`。`scripts/audit_expanded.py` 保存数据质量证据，`scripts/compare_checkpoints.py` 在同一冻结测试集比较新旧模型。`--balance-tasks` 只改变训练采样权重，不增加独立数据数量；验证轮次按各任务 NLL 的平均值选择。

概率目标恰为 `[0.5, 0.5]` 时仍参与 NLL/Brier/ECE，但不纳入唯一目标众数准确率；报告单独给出 `ambiguous_count` 和高置信度接受的模糊样本数，避免把不确定问题硬算成“是”。

## 数据与训练

底座采用 [HFL 的 `rbt3`](https://huggingface.co/hfl/rbt3)，是公开的三层中文 RoBERTa 编码器。共享编码器和打分头都参与训练，不依赖关键词规则决定答案。

数据覆盖职场、家庭、朋友、伴侣、服务、邻里、线上和协作八个领域，每个领域 10 个场景，每个场景构造 Choice、Score、Noul 三个任务。

| 集合 | 场景组 | 决策样本 | 用途 |
| --- | ---: | ---: | --- |
| train | 56 | 168 | 更新神经网络参数 |
| validation | 8 | 24 | 按 NLL 选择最佳轮次、早停 |
| calibration | 8 | 24 | 只拟合温度参数 |
| test | 8 | 24 | 最终报告，不参与选轮次和调温度 |

同一场景的三个任务始终留在一个集合；加载时检查跨集的场景组、相同场景和样本 ID。`data/seed_cases.json` 是原创合成种子，`scripts/build_data.py` 按固定随机种子和领域分层生成 JSONL。`data/demo/manifest.json` 保存来源、划分和内容哈希。脚本拒绝覆盖已有数据。

**这 240 条是小规模演示数据，尚未经多人标注。** 软标签是作者给定的教学目标，不是实际人群赞同频率。模型可能学到措辞线索；当前测试集只有 8 个独立场景组，指标只能证明这次实验的表现，不能代表真实世界的“情商”或 Jev 能力。

默认参数：20 轮上限、batch size 12、学习率 3e-5、AdamW、梯度裁剪 1.0、验证集早停 patience 5、最大长度 256。CUDA 支持 BF16 时启用 BF16，否则使用 FP32。训练保存底座版本、数据 SHA256、超参数、软件版本和实际设备；GPU 运算仍可能有细微非确定性。

较长输入会报错，避免静默截断导致遗漏人际上下文。对于适合此底座的文本，可在训练时使用 `--max-length 512`，但会增加资源需求。多个问题以有限大小的候选批次计算，不宣称增加问题数时延不变。

## 训练产物

每个训练目录包含：

- `encoder/`、`tokenizer/`、`head.safetensors`：可独立离线加载的已训练权重。
- `metadata.json`：来源、数据哈希、参数和环境。
- `history.json`：每轮训练损失和验证表现。
- `calibration.json`：独立温度校准前后指标。
- `evaluation.json`：均匀基线、未训练打分头基线、训练后原始/校准指标，按任务和领域拆分。
- `test_predictions.json`：逐条测试样本、目标分布和预测概率，可直接检查错误。

评估包含目标众数准确率、软标签 NLL、Brier、软标签 ECE、阈值覆盖率和被接受样本的准确率。ECE 比较最大预测概率和目标分布对所选候选的支持度；不是基于真实人群频率的校准证明。没有接受样本时接受准确率为 `null`。

## 换成自己的数据

参见 [data/ANNOTATION.md](data/ANNOTATION.md)。准备相同格式的四个 JSONL 文件后：

```powershell
.venv\Scripts\python.exe -m renqing.cli validate --data data\my-corpus
.venv\Scripts\python.exe -m renqing.cli train --data data\my-corpus --output runs\my-corpus-v1
```

项目将得体沟通定义为：能理解已有证据、兼顾需求、表达必要边界、减少不必要冒犯、尊重拒绝与隐私；具体场景有多种合理答案时，使用分布标注保留分歧。真正提高泛化能力需要增加独立场景和人工标注，而不是反复训练这 80 个场景。

原始底座使用 Apache-2.0；详见模型卡和 `BASE_MODEL_NOTICE.md`。本项目原创代码与示例同样按 Apache-2.0 提供。
