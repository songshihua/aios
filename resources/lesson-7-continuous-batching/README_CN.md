# Lesson 7：Continuous Batching（连续批处理）

## 本节目标

把调度器从一次性批处理升级为**持续运转的循环**：维护两个集合——**pending 队列**
（等待 prefill 的新请求）和 **running 集合**（正在 decode 的请求）。每个迭代只做一件事：
admit 一个 pending 请求做 prefill，或者对整个 running 集合跑一次 decode。
**一旦请求结束，立即释放它的页和 slot**，下一轮迭代就能把刚空出来的资源分给新的 pending 请求。

这就是 **continuous batching**。它把调度器从"批处理"变成了"服务器"，并且是后续课程
（CUDA Graphs、Prefix Caching、API Server）的前提。

![Continuous Batching swim lanes](continuous_scheduling.png)

> 上图：4 个请求 (out_len A:2, B:5, C:3, D:2) 在 `max_running=2` 下的调度时间线。
> 每行是一个请求的完整生命周期——灰色 pending → 橙色 prefill → 绿色 decode →
> 黄色 idle (running 中但本 iter 在跑别的 prefill) → 灰虚线 freed。
> 蓝色弧线箭头标出 slot 在请求间的复用：A 一释放 C 立刻顶上，C 一释放 D 立刻顶上。

## 和 Static Batching 的对比

同样的 workload 跑在 Lesson 6 的 static batching（`max_running=2` → 拆成 2 个 wave）：

![Static Batching waves](static_scheduling.png)

关键区别：

- **Wave 边界硬切**：C, D 在 iter 0–5 完全不存在于调度器中（灰色 deferred），必须等 wave 1 finalize 才能进 wave 2
- **Wave 内 WASTED 仍在**：A 在 iter 1 完成但 slot 锁到 iter 5 finalize；D 同理在 iter 7→9
- **总耗时**：static-waves = 10 iter vs continuous = 8 iter（同硬件、同 cap）
- **slot 复用**：static 不跨 wave，continuous 跨请求

## 与 mini-sglang 对齐

Lesson 7 的结构直接对应 mini-sglang 的调度器分层：

```
scheduler/
├── scheduler.py   Scheduler           ← 顶层编排（prefill-first 策略 + 资源释放）
├── prefill.py     PrefillManager      ← pending_list + add_one_req + schedule_next_batch
├── decode.py      DecodeManager       ← running_reqs + filter_reqs + schedule_next_batch
├── table.py       TableManager        ← 未改动
└── cache.py       CacheManager        ← 未改动
```

方法命名、职责划分与 mini-sglang 保持一致：`add_one_req` / `schedule_next_batch` /
`filter_reqs` / `inflight_tokens` / `runnable` / `_free_req_resources`。

## 刻意简化（会在后续课程中逐步补齐）

| 简化点 | 原因 | 哪节课补齐 |
|---|---|---|
| **bsz=1 prefill**（每步只 admit 一个 prompt） | `_batched_paged_attention` 的 prefill 分支需要同一批内所有 prompt 长度一致；若要多请求 varlen prefill 就必须换 FlashAttention varlen。 | Lesson 8（FlashAttention + 扁平 token 表示） |
| **不支持 chunked prefill** | 长 prompt 分段 prefill 会引入 `ChunkedReq` 这层抽象，超出了本节的"scheduler 重写"主题。 | 后续长上下文专题 |
| **不混合 prefill + decode batch** | mini-sglang 默认也不混合（prefill-first），无需额外改造 Batch 形状。 | 不再作为独立专题 |
| **不做 prefix caching** | `CacheManager.cache_req` 继续保持注释状态，释放走直接的 `_free(pages)`。 | Lesson 13（Prefix Caching） |
| **单 CUDA stream，无 overlap 调度** | `overlap_loop` 属于高级优化。 | 后续专题 |
| **单进程，不走 IPC** | `receive_msg` / `run_forever` 等服务器循环是 Lesson 15 的主题。 | Lesson 15（API Server） |

## 工作流（一图胜千言）

![Request state machine](code_architecture.png)

每个请求都流过 **三个集合**：

- **PrefillManager.pending_list** — 入口队列，只持 `_PendingReq`，不占任何 GPU 资源
- **DecodeManager.running_reqs** — 活跃集合，持 `_ReqState{ Req(table_idx, generated[]), sampler, finished }`，绑定 KV pages
- **Scheduler.finished** — 出口队列，资源已归还，仅保留 `req.generated` 输出 token

中间两个 panel（**ADMIT** / **FREE**）展示了每次转换的具体代码步骤和数据结构变化：

| 转换 | 触发 | 关键步骤 | 类型变化 |
|---|---|---|---|
| ADMIT | `prefill_manager.schedule_next_batch` | pop pending head → `table.allocate()` + `cache.allocate(N)` → 写 `token_pool` → 包一层 `_ReqState` → `decode.add_req` | `_PendingReq → _ReqState` |
| FREE | `process_batch_output` 的 decode 分支 | `advance` 每个 running → `filter_reqs()` 摘走 finished → `cache._free(pages)` + `table.free(slot)` → `finished.append` | `_ReqState`（带资源）→ `_ReqState`（资源已释放，generated 保留）|

## 运行与验证

```bash
python resources/lesson-7-continuous-batching/run_lesson7.py \
    --model Qwen/Qwen3-0.6B \
    --num-seqs 64 --prompt-len 256 --out-low 32 --out-high 256 \
    --max-running 32
```

## 代码路径速查

- 顶层编排：`python/aios/scheduler/scheduler.py`（`Scheduler`）
- pending 队列与 bsz=1 admission：`python/aios/scheduler/prefill.py`
- running 集合与 decode 批：`python/aios/scheduler/decode.py`
- 用户入口：`python/aios/llm/llm.py::LLM.generate`
- 共用类型：`python/aios/scheduler/common.py`（`ScheduledBatch`、`_ReqState`、`_PendingReq`）
- 数据结构：`python/aios/core.py`（`Req.generated: List[int]`）

## 下一节预告

Lesson 7 把 prefill 限制在 bsz=1，是为了绕开 `_batched_paged_attention` 的同长度约束。
**Lesson 8 会引入 FlashAttention varlen**：attention 接受扁平的 `(total_tokens, heads, dim)`
张量 + `cu_seqlens`，于是同一次 prefill 就能同时处理多个不同长度的 prompt——彻底释放
continuous batching 在 prefill 侧的吞吐。
