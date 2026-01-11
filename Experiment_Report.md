# vLLM Routing Experiment - Discovery & Analysis Log

## 1. 当前实验发现 (Experiment Findings)

### 1.1 数据分布与负载特性
*   **双模态延迟分布 (Bimodal Latency)**: 
    *   **计算密集区 (Compute-bound)**: 在低负载下（RPS < 64），TTFT 主要由 Prefill 时间决定。延迟在 0.05s - 0.3s 之间波动。此阶段 TTFT 与 `input_len` 呈弱线性关系，受系统噪声影响大，预测难度高，但预测价值低（因为都在 QoS 范围内）。
    *   **排队密集区 (Queue-bound)**: 在高负载下（RPS ≥ 64），TTFT 出现指数级增长，甚至达到 50s-100s。这表明请求在 vLLM 内部的 `Scheduler` 队列或系统网络栈中长时间积压。

### 1.2 现有预测模型的局限性
*   **$R^2$ 的欺骗性**: 虽然引入排队论特征（`reqs_sq`）后 $R^2$ 达到 0.99，但这主要是模型拟合了高负载下巨大的排队延迟数值。
*   **掩盖效应**: 由于 MSE 损失函数对大误差敏感，模型过度关注如何预测 50s vs 60s 的差异，而忽略了区分 0.2s vs 0.5s 的能力。
*   **状态不可见**: 目前仅使用客户端视角的 `active_reqs` 作为特征。但 vLLM 内部状态（Running vs Waiting）对延迟影响截然不同。客户端看到的 100 个并发请求，可能意味着 20 个在跑（快），80 个在排队（极慢）。简单的计数特征无法区分这种微观状态。

### 1.3 异常现象分析
*   **404 / Connection Issues**: 
    *   观察到在高并发下出现 404 或连接重置。
    *   **推测原因**: 并非 vLLM 拒绝排队超过 100 个请求（vLLM 会排队），而是 HTTP Server (如 `uvicorn`/`fastapi`) 或 OS 网络栈层面的超时/溢出。
    *   **长尾延迟**: 部分请求耗时极长，表明请求进入了“死等待”状态，直到被调度。从 Router 发出请求到 vLLM 真正开始处理（Scheduler Dequeue），中间存在巨大的 Gap。

---

## 2. 改进设计建议 (Proposed Redesign)

### 2.1 从“黑盒”转向“灰盒” (Grey-box Routing)
*   **获取真实状态**: 
    *   Router 不应仅靠猜（Client-side metrics）。
    *   利用 vLLM 的 `/metrics` 端点（基于 Prometheus 格式）或 `/health` 端点。
    *   **关键指标**:
        *   `vllm:num_requests_running`: 当前 GPU 上并发处理的数量。
        *   `vllm:num_requests_waiting`: 调度队列中的数量（排队延迟的直接来源）。
        *   `vllm:gpu_cache_usage_perc`: KV Cache 占用率（预测即将发生的 Swap/排队）。

### 2.2 数据采集与探索性分析 (EDA Strategy)
*   **目标**: 建立更精细的 `System State -> Performance` 映射。
*   **新采集方案**:
    *   Router 增加后台线程，以高频（如 10Hz）拉取各后端的 `/metrics`。
    *   将 `metrics` 快照与每个 Request 的 `TTFT` 记录关联。
*   **分析重点**:
    *   `num_requests_waiting` > 0 时的 TTFT 分布。
    *   `gpu_cache_usage` 接近 100% 时的性能拐点。
    *   寻找 **"Knee Point" (拐点)**: 并不是要预测 100s 的延迟，而是要在延迟从 0.5s 变成 2s 的那一刻识别出来，并进行流量转移。

### 2.3 分层建模与归一化
*   **解决大数值干扰**: 
    *   不要直接预测 TTFT 秒数。
    *   **方案 A (分类)**: 预测 System Load Level (Low, Medium, High/Overloaded)。
    *   **方案 B (加权/分段回归)**: 针对不同的负载区间训练不同的模型，或者使用 Log-transform (对目标变量取对数 `log(TTFT)`) 来压缩大数值的影响，让模型关注相对误差而非绝对误差。

### 2.4 路由策略调整
*   **QoS 优先**: 
    *   如果预测到某节点排队长度 > 阈值（或预测延迟 > 5s），Router 应直接熔断该节点（Circuit Breaker），不再发送新请求，直到积压清除。
    *   避免在“两个都很慢的节点”中做选择题，而是应该进行 **Admission Control (准入控制)**，直接返回 HTTP 503 或排队在 Router 侧。

---

## 3. 灰盒数据分析与分层路由策略 (Phase 2 Analysis & Strategy Update)

### 3.1 真实 Metrics 探索分析 (EDA on Grey-box Data)
通过对 `profile_system.py` 的改造，我们成功采集了包含真实 `vllm:num_waiting` 和 `vllm:num_running` 的数据集。分析结果揭示了几个关键事实：

1.  **Waiting 队列之谜**: 
    *   在高延迟（TTFT > 20s）的样本中，vLLM 报告的 `num_waiting` 竟然全是 **0.0**。
    *   这推翻了最初的假设（请求在 Engine 内部排队）。
    *   **结论**: 拥塞发生在 vLLM 之前的 HTTP 层或应用层信号量限制。请求在还没进入 Engine 之前就已经被卡住了。这也解释了为什么 `num_waiting` 与 TTFT 的相关性极低 (0.07)，而 Router 端的 `active_reqs_client` 相关性极高 (0.98)。

2.  **性能的三重奏 (Three Performance Zones)**:
    我们根据 TTFT 将系统状态划分为三个清晰的区域，每个区域的特征截然不同：

    *   **Zone 1: Low Load (TTFT < 1s)**
        *   **特征**: 噪声主导。TTFT 与任何负载指标（Running/Waiting/Active）的相关性都很低甚至为负。
        *   **含义**: 此时系统资源充足，路由策略的边际收益极低。任何简单的负载均衡（Round Robin / Least Connection）都是最优解。
        *   **策略**: **Least Load (Baseline)**。

    *   **Zone 2: Medium Load (1s <= TTFT < 10s)**
        *   **特征**: **可预测性最强**。`num_running`（正在运行的并发数）与 TTFT 呈现显著正相关 (**R=0.66**)。
        *   **含义**: 这是 vLLM 全速运转但未饱和的阶段。此时，每个新加入的请求都会与现有的 `num_running` 个请求竞争 GPU 资源。
        *   **策略**: **Predictive Routing (MLP/Regression)**。这是预测模型发挥最大价值的区间。我们需要训练一个模型，输入 `(num_running, input_len)`，输出预测的 TTFT，从而在多个尚未饱和的节点中选择最优者。

    *   **Zone 3: High Load / Saturation (TTFT >= 10s)**
        *   **特征**: 线性拥塞。`active_reqs_client` 与 TTFT 相关性高达 **0.99**，且 `num_running` 锁定在 100 (Hard Limit)。
        *   **含义**: 系统已达吞吐极限，进入纯排队模式。任何新请求都只是在队尾增加等待时间。
        *   **策略**: **Congestion Control (Hard Limit)**。此时不应预测“谁慢得少一点”，而应直接熔断或拒绝服务，直到负载回落。

### 3.2 提出的分层路由策略 (Hierarchical Routing Strategy)

基于以上发现，我们放弃“一个模型预测所有场景”的想法，转而采用分层策略：

1.  **Layer 1: Safety Guard (Congestion Control)**
    *   **阈值**: `active_reqs_client > 100` (基于数据的 High Latency 拐点)。
    *   **动作**: 熔断。将该后端权重置为 0，防止雪崩效应。

2.  **Layer 2: Performance Optimizer (Predictive Model)**
    *   **适用范围**: `active_reqs_client <= 100` 且系统处于 Zone 2。
    *   **模型**: 仅使用 Zone 2 的数据训练一个 `RandomForest` 或 `MLP`。
    *   **特征**: `num_running` (服务端真实并发), `input_len`。
    *   **动作**: 预测 TTFT，选择预测值最小的后端。

3.  **Layer 3: Fallback (Least Load)**
    *   **适用范围**: Zone 1 (Low Load) 或 模型置信度低时。
    *   **动作**: 直接选择 `active_reqs_client` 最小的节点。

### 3.3 下一步行动计划
1.  **数据清洗与模型训练**:
    *   从 `profiling_data.csv` 中提取 Zone 2 数据 (1s <= TTFT < 10s)。
    *   训练一个轻量级预测模型 (`models/zone2_predictor.pkl`)。
2.  **网关逻辑实现**:
    *   修改 `gateway.py`，实现上述三层逻辑。
    *   集成 `metrics` 拉取线程，为 Zone 2 模型提供 `num_running` 特征。
