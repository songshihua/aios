# Lesson 11：Batch Sampling（Temperature / Top-k / Top-p）

本课在 Lesson 10 CUDA Graphs 之后，把最后一个仍由 Python 逐请求执行的 GPU 阶段——采样——改造成与 mini-sglang 一致的批量采样路径。

## 1. 背景：模型算完 logits，为什么还没有结束

模型 forward 的输出不是文字，也不是最终 token，而是词表中每个 token 的未归一化分数：

```text
logits: [batch_size, vocab_size]
                    │
                    ├─ greedy / temperature
                    ├─ top-k / top-p filter
                    └─ random sampling
                              │
                              ▼
next_token_ids: [batch_size]
```

Lesson 10 已经能用 CUDA Graph replay decode forward，但旧采样器仍然对 batch 中每个请求执行一次 Python 循环：

```python
next_tokens = []
for i, req in enumerate(batch.reqs):
    token = Sampler(req.sampling_params).sample(logits[i : i + 1])
    next_tokens.append(token)
```

这条路径有三个问题：

1. batch 越大，Python 循环和 kernel launch 次数越多；
2. 每步都重复构造 `Sampler`，参数所有权不清晰；

Sampling 的首要目标是生成质量和可控性，本课不承诺提升模型 forward 吞吐；但批量化可以避免采样阶段随 batch size 线性增加 host 调度开销。

## 2. 原理：从 logits 到 token

### 2.1 Greedy sampling

Greedy 直接选择最大 logit：

```text
token = argmax(logits)
```

它没有随机性，适合 benchmark、回归测试和要求稳定输出的任务。AIOS 中以下配置走 greedy：

```python
SamplingParams(temperature=0.0)
SamplingParams(top_k=1)
```

当整个 batch 都是 greedy 时，`BatchSamplingArgs.temperatures` 设为 `None`，Sampler 直接执行一次 batched `argmax`，不计算 softmax。

### 2.2 Temperature

temperature 在 softmax 前缩放 logits：

```text
p_i = softmax(logit_i / T)
```

- `T < 1`：分布更尖锐，更偏向高分 token；
- `T = 1`：保持原分布；
- `T > 1`：分布更平坦，随机性更强；
- `T = 0`：工程上不用除法计算，而是切换到 greedy。

混合 batch 可能同时包含 greedy 和随机请求。为了保持一次批量 kernel 调用，greedy 行的 temperature 被替换为 `1e-6`，分布近似 one-hot；只有全 greedy batch 才使用独立的 `argmax` 快路径。

### 2.3 Top-k

Top-k 只保留概率最高的 k 个 token，再在它们之间采样：

```text
原候选： [A, B, C, D, E, ...]
top_k=3： [A, B, C]
```

它直接限制候选数量。`top_k=-1` 表示禁用；批量参数中会被转换成 `vocab_size`，这样每一行仍可放进同一张 tensor。

### 2.4 Top-p（nucleus sampling）

Top-p 按概率从大到小累加，只保留累计概率达到阈值 p 所需的最小候选集合：

```text
token       A     B     C     D
prob       .50   .25   .15   .10
cumsum     .50   .75   .90  1.00

top_p=.80  -> 保留 A、B、C
```

它会随分布动态调整候选数量：模型很确定时集合较小，模型不确定时集合较大。`top_p=1.0` 表示禁用。

同时设置 top-k 和 top-p 时，本课沿用 FlashInfer `top_k_top_p_sampling_from_probs` 的默认顺序：先应用 top-k，再应用 top-p。

### 2.5 Multinomial、Gumbel-max 与 FlashInfer

对离散概率分布采样可以用 `torch.multinomial`，也可以对每个候选加入 Gumbel 噪声后取最大值：

```text
token = argmax(log(p) + g),  g ~ Gumbel(0, 1)
```

Gumbel-max 给出了“随机采样可以转化为并行 argmax”的重要视角。它与从 categorical distribution 中抽样具有相同分布，但显式生成整个词表大小的噪声并不一定是最高效的生产实现。

AIOS 与当前 mini-sglang 一样，使用 FlashInfer sampling kernels。FlashInfer 负责 softmax、top-k/top-p 和抽样，课程代码不显式物化 Gumbel tensor。

### 2.6 为什么参数必须是 batch tensor

连续批处理中的请求可以有不同策略：

```text
req 0: greedy
req 1: temperature=0.7
req 2: temperature=0.8, top_k=20
req 3: temperature=0.9, top_k=30, top_p=0.9
```

`BatchSamplingArgs` 把它们整理成 GPU tensor：

```text
temperatures = [1e-6, 0.7, 0.8, 0.9]
top_k       = [vocab, vocab, 20, 30]
top_p       = None 或 [1.0, 1.0, 1.0, 0.9]
```

只有 batch 中确实启用了某种过滤时才创建对应 tensor。例如所有请求都禁用 top-p 时，`top_p=None`，Sampler 直接选择不包含 top-p 的 kernel 路径。

## 3. 具体实现

### 3.1 `SamplingParams`：每请求策略的来源

`python/aios/core.py` 已在前面课程中定义：

```python
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

`SamplingParams` 属于 `Req`，生命周期与请求一致。Sampler 不保存某个请求的参数，只在调度一个 batch 时读取它们。

### 3.2 `BatchSamplingArgs`：一次 forward 的采样元数据

`python/aios/engine/sample.py` 新增：

```python
@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
```

它与 `Batch` 类似，描述的是一次 forward，而不是全局配置。三个 tensor 的第一维都对应 `batch.reqs` 的真实请求顺序，不包含 CUDA Graph padding 使用的 dummy request。

### 3.3 `Sampler.prepare()`：CPU 参数整理成 GPU tensor

核心步骤如下：

```python
params = [req.sampling_params for req in batch.reqs]
if all(param.is_greedy for param in params):
    return BatchSamplingArgs(temperatures=None)

temperatures = [...]
top_ks = [...]
top_ps = [...]
```

随后通过 pinned CPU tensor 和 non-blocking copy 搬到当前 device：

```python
torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
```

这与 mini-sglang 的 `make_device_tensor()` 一致。当前 AIOS 是单 stream 顺序执行；mini-sglang 还能在 scheduler stream 上准备下一批 metadata，并与上一批 engine 计算重叠，这项 overlap scheduling 不属于本课。

### 3.4 `sample_impl()`：选择最小必要 kernel 路径

先用 FlashInfer 完成 temperature softmax：

```python
probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported)
```

再根据可选参数选择路径：

```text
top_k=None, top_p=None  -> sampling_from_probs
top_k!=None             -> top_k_sampling_from_probs
top_p!=None             -> top_p_sampling_from_probs
两者都有                -> top_k_top_p_sampling_from_probs
```

Hopper（SM90+）上启用 PDL；其他 GPU 自动关闭，不硬编码架构。

### 3.5 Scheduler：采样参数与 batch 一起交给 Engine

`ForwardInput` 增加 `sample_args`：

```python
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D
    write_tuple: Indice2D
```

`Scheduler._prepare_batch()` 的顺序为：

```text
pad CUDA Graph batch
  -> allocate KV pages
  -> positions / input / write mapping
  -> attention metadata
  -> sampler.prepare(batch)
  -> ForwardInput
```

这个字段、所有权和调用时机与 mini-sglang 一致。Scheduler 决定 batch，因此也负责把请求级参数整理成 batch 级 metadata；Engine 只消费准备好的输入。

### 3.6 Engine：一个 batch 只调用一次 Sampler

Engine 初始化时长期持有：

```python
self.sampler = Sampler(device, model_config.vocab_size)
```

forward 后只保留真实请求的 logits，并批量采样：

```python
next_tokens = self.sampler.sample(
    logits[: batch.size], args
).to(torch.int32)
```

`logits[:batch.size]` 非常重要。CUDA Graph decode 可能把 3 个真实请求 pad 到 4，但 dummy request 只用于稳定 forward shape，不能进入采样、EOS 判断或资源回收。

### 3.7 与 CUDA Graph 的边界

本课没有把随机采样捕获进 CUDA Graph：

```text
CUDA Graph replay: model forward -> logits
                                      │
                                      ▼
eager GPU path:                 batch sampling
                                      │
                                      ▼
Scheduler:                     append / finish
```

原因是 model forward 拓扑固定、适合 replay；sampling 参数随请求和 batch 改变。将 sampling 保持在 graph 外，既保留 Lesson 10 的收益，也让不同请求自由选择策略。

### 3.8 与 mini-sglang 的对齐和差异

| 项目 | AIOS Lesson 11 | mini-sglang |
|---|---|---|
| `SamplingParams` 字段与 greedy 判断 | 一致 | 基准 |
| `BatchSamplingArgs` 字段 | 一致 | 基准 |
| `Sampler.prepare(batch)` | 一致 | 基准 |
| FlashInfer 四条 kernel 分支 | 一致 | 基准 |
| Scheduler 准备 sampling args | 一致 | 基准 |
| Engine 批量采样 | 一致 | 基准 |
| overlap scheduling | 暂无，单 stream | scheduler/engine 双 stream |
| token D2H | Scheduler 中同步复制 | Engine non-blocking copy + event |
| tensor parallel | 暂无 | rank 0 采样并协调 |

后面三项是尚未进入当前课程的并发与多 GPU能力，不改变本课采样算法和数据语义。

## 4. 验证结果

### 4.1 编译和导入检查

```bash
python -m compileall -q python/aios
PYTHONPATH=python python -c \
  "from aios.engine import BatchSamplingArgs, Sampler; print('Import OK')"
```

预期：命令成功退出并打印 `Import OK`。

### 4.2 Sampler 正确性检查

不加载模型，只构造 logits：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python \
python resources/lesson-11-sampling/run_lesson11.py \
  --suite check
```

覆盖：

- 全 greedy batch 的 `argmax` 快路径；
- greedy、temperature、top-k、top-p 混合参数打包；
- top-k 抽样结果不会越过候选集合；
- 输出 shape、dtype 和 token 范围。

RTX 3090 实测：

```text
[CHECK] all-greedy fast path: PASS
[CHECK] mixed per-request arguments: PASS
[CHECK] top-k support constraint: PASS
[CHECK] top-p support constraint: PASS
```

### 4.3 旧逐请求采样与新批量采样

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python \
python resources/lesson-11-sampling/run_lesson11.py \
  --suite bench --batch-size 32 --vocab-size 151936 --iterations 50
```

Runner 将 Lesson 10 的 Python per-request `softmax + multinomial` 与本课 FlashInfer batch sampler 放在相同 logits 上测量。JIT/warmup 不计入计时。

RTX 3090、FlashInfer 0.5.3 实测：

| 路径 | 单次 batch 延迟 | 采样 token/s | 加速比 |
|---|---:|---:|---:|
| Lesson 10 per-request | 6.223 ms | 5,142.40 | 1.00× |
| Lesson 11 batch FlashInfer | 0.164 ms | 195,442.76 | 38.01× |

这是只测 sampling kernel 和 Python 调度的 microbenchmark，不代表整个模型生成会加速 38 倍。E2E 中 model forward 仍占主要时间；实际收益取决于 batch size、词表大小和 GPU。

### 4.4 模型 E2E 与 CUDA Graph 兼容性

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
FLASHINFER_CACHE_DIR=/tmp/flashinfer-aios-lesson11 \
PYTHONPATH=python \
python resources/lesson-11-sampling/run_lesson11.py \
  --model /data4/home/yan.wang/huggingface/Qwen3-0.6B \
  --cuda-visible-devices 1 --suite e2e --cuda-graph
```

E2E batch 同时包含 greedy、top-k、top-k + top-p 请求，并检查：

- 每个请求都生成指定数量 token；
- 两个相同 greedy 请求得到相同结果；
- CUDA Graph padding 后只有真实请求参与采样；
- 生成循环、EOS/长度判断和 KV cache 回收保持正常。

Qwen3-0.6B、RTX 3090 实测：

```text
[E2E] mixed sampling with cuda_graph=True: PASS lengths=[8, 8, 8, 8]
```

### 4.5 本课结论

```text
Lesson 10: batched logits -> Python loop -> per-request sampling
Lesson 11: batched logits -> BatchSamplingArgs -> one vectorized sampler
```

完成本课后，AIOS 的请求级采样策略仍然彼此独立，但执行已经变成 batch 级 GPU 数据流。下一课 Prefix Caching 可以继续优化 prefill，而无需再修改采样与 decode forward 的职责边界。
