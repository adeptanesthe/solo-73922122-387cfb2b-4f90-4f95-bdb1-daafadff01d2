# 流式语种分段服务（Streaming Language Segmenter）

把流式语种识别器逐帧输出的后验概率，整理成**稳定、可回放、可封存**的语种区间；
并提供 **HMM 转移矩阵校准**接口，用人工校对过的录音估计各语种间的切换倾向。
技术栈：FastAPI + NumPy，Docker Compose 启动。

- 会话创建时设定 **2～4 个语种**、帧长 `frame_ms`、切换所需连续帧数
  `switch_frames`、后验阈值 `posterior_threshold`（以及可选的切换惩罚
  `switch_penalty`）。
- 按全局序号 `start_seq` 提交帧块；服务用**带切换惩罚的 HMM 平滑状态**输出
  已确认区间。
- 只有新语种**连续达到阈值帧数**才切段；低置信帧沿用上一状态；单帧抖动
  不会切段。
- 完全重复的块返回**原结果**且不改变会话；迟到 / 越界 / 冲突块**一律拒绝**
  且不改变会话。
- 提交 `final: true`（或调用 `/seal`）后**封存**区间，之后不再接受新块。
- `POST /calibration`：提交 2～4 个语种的训练/留出录音（合计 ≤ 1000 帧，
  训练可缺标、留出须标全），从切换惩罚 2 的矩阵起步做 5 轮带标签约束的 EM，
  按留出集真实序列条件对数概率决定是否启用；**候选严格更高才替换**。
- 新会话可选用同语种组**当时已启用**的校准矩阵（按请求语种顺序重排）；
  已开会话保留建会话时的矩阵，校准期间收块不中断。

---

## 1. 快速开始

```bash
docker compose up --build
```

服务监听 `http://localhost:8000`，交互式文档在 `/docs`。

本地开发/自测（无需 Docker）：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q                     # 31 个单元/接口用例（另 1 个 live 用例按需启用）
uvicorn app.main:app --port 8000
python examples/demo.py http://127.0.0.1:8000
python examples/calibration_demo.py http://127.0.0.1:8000
# 可选：真实 HTTP 端到端测试
SEGMENTER_BASE_URL=http://127.0.0.1:8000 python -m pytest tests/test_live.py -q
```

Docker 下跑测试（compose 的 `test` profile，会等 `segmenter` healthy 后执行）：

```bash
docker compose --profile test run --rm tests
```

`examples/demo.py` 只依赖标准库，演示了创建会话、正常分块、重复块、迟到块、
越界块、低置信尾帧、封存与封存后提交。`examples/calibration_demo.py` 演示
一次完整校准（含缺标训练录音）、第二次校准对照、矩阵查询/重排、新会话选用
校准矩阵与未校准语种组的 409。

---

## 2. API

### 2.1 创建会话

`POST /sessions`

```json
{
  "languages": ["zh", "en", "ja"],
  "frame_ms": 20,
  "switch_frames": 3,
  "posterior_threshold": 0.55,
  "switch_penalty": 2.0
}
```

| 字段 | 约束 | 说明 |
|---|---|---|
| `languages` | 2～4 个、非空、去重（忽略大小写） | 顺序即每帧后验向量的列顺序 |
| `frame_ms` | 1～1000 | 单帧时长，用于换算毫秒区间 |
| `switch_frames` | 1～1000 | 建段/切段所需的**连续**帧数 |
| `posterior_threshold` | (0, 1] | 最大原始后验低于该值的帧视为低置信 |
| `switch_penalty` | [0, 50] | HMM 切换到其它语种的对数代价（纳特），越大越"粘" |

响应 `201`：

```json
{
  "session_id": "c5b6b0b8…",
  "languages": ["zh", "en", "ja"],
  "frame_ms": 20,
  "switch_frames": 3,
  "posterior_threshold": 0.55,
  "switch_penalty": 2.0
}
```

### 2.2 提交帧块

`POST /sessions/{session_id}/chunks`

```json
{
  "start_seq": 8,
  "posteriors": [[0.02, 0.96, 0.02], [0.02, 0.95, 0.03]],
  "final": false
}
```

- `start_seq`：本块第一帧的**全局序号**，从 0 开始，必须等于 `next_seq`。
- `posteriors`：形状 `(帧数, 语种数)`，每行非负、和 > 0（内部会归一化）。
- `final`：置 `true` 等价于在本块处理成功后立即封存会话。

响应 `200`：

```json
{
  "session_id": "…",
  "start_seq": 8,
  "end_seq": 14,
  "accepted": true,
  "duplicate": false,
  "sealed": false,
  "segments": [
    {"language": "zh", "start_frame": 0, "end_frame": 7,
     "start_ms": 0, "end_ms": 160, "duration_ms": 160}
  ],
  "open_segment": {"language": "en", "start_frame": 8, "end_frame": 14,
                   "start_ms": 160, "end_ms": 300, "duration_ms": 140},
  "current_language": "en",
  "pending_language": null,
  "pending_frames": 0,
  "next_seq": 15
}
```

- `segments`：**已关闭**区间（一旦关闭不会再变），闭区间 `start_frame..end_frame`。
- `open_segment`：已通过阈值确认、仍在延伸的当前区间（快照）。
- `pending_language` / `pending_frames`：尚未达到阈值的候选切换进度。
- `next_seq`：下一块的期望 `start_seq`。

### 2.3 其它端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/sessions/{id}` | 会话完整状态（参数、当前段、候选切换、`next_seq`、生效矩阵） |
| `POST` | `/sessions/{id}/seal` | 封存；未闭合尾段按最后一帧封口，返回全部区间 |
| `GET` | `/sessions/{id}/segments` | 当前（或封存后的）区间列表 |
| `GET` | `/healthz` | 健康检查 |

### 2.4 校准 HMM 转移矩阵

`POST /calibration`

```json
{
  "languages": ["zh", "en", "ja"],
  "train": [
    {
      "posteriors": [[0.45, 0.30, 0.25], [0.42, 0.33, 0.25]],
      "labels": ["zh", null]
    },
    { "posteriors": [[0.28, 0.44, 0.28]], "labels": null }
  ],
  "held_out": [
    {
      "posteriors": [[0.45, 0.30, 0.25], [0.27, 0.46, 0.27]],
      "labels": ["zh", "en"]
    }
  ]
}
```

请求约束：

| 字段 | 约束 |
|---|---|
| `languages` | 2～4 个、非空、去重（忽略大小写），顺序即后验列顺序 |
| `train` | 至少 1 条录音；`labels` 可整体省略、为 `null`，或逐帧 `null`（缺标） |
| `held_out` | 至少 1 条录音；**必须逐帧标全**，任一 `null` 返回 422 |
| 帧预算 | 训练 + 留出**合计最多 1000 帧**，超出返回 422 `frame_budget_exceeded` |
| 后验 | 每行非负、有限、行和 > 0；标签必须出自 `languages` |

算法（实现见 `app/calibration.py`）：

1. **起步矩阵**：切换惩罚 2 的矩阵——保持 0 对数代价、切换 -2 纳特，逐行
   logsumexp 归一化，与在线会话构造规则完全相同。
2. **5 轮 EM**：每条录音首帧使用**均匀语种先验**（与在线会话一致）；已标
   帧把隐状态硬约束为人工语种，缺标帧全部状态开放；前向-后向统计**所有
   录音、所有相邻帧**的期望转移次数 `ξ_t(i,j)`。
3. 每轮把期望计数矩阵**每种转移加 1 个伪计数**后按行归一化，得到下一轮矩阵。
4. **留出打分**：对每条留出录音计算真实语种序列的条件对数概率
   `log P(z|x) = log P(x,z) − log P(x)`（证据项由前向算法得到），各录音
   得分相加；训练录音不参与打分。
5. **启用判定**：候选分数**严格高于**对照基线才替换同语种组的在用矩阵；
   相等或更低则不启用。首次校准以惩罚 2 矩阵为对照；之后以同组**当时已
   启用**的矩阵为对照（按本次语种顺序重排）。

响应 `200`（节选）：

```json
{
  "languages": ["zh", "en", "ja"],
  "group": ["en", "ja", "zh"],
  "baseline_score": -30.47,
  "candidate_score": -19.06,
  "score_delta": 11.41,
  "enabled": true,
  "replaced": false,
  "active_source": "candidate",
  "rounds": 5,
  "switch_penalty_start": 2.0,
  "train_frames": 480,
  "held_out_frames": 240,
  "total_frames": 720,
  "baseline_matrix": [[0.787, 0.107, 0.107], ["..."]],
  "candidate_matrix": [["..."]],
  "active_matrix": [["..."]],
  "calibration_count": 1,
  "enabled_at": "2026-11-22T…Z"
}
```

- `baseline_score` / `candidate_score`：对照矩阵与候选矩阵在留出集上的
  条件对数概率之和；`score_delta` 为两者之差。
- `enabled=false` 时 `active_matrix` 等于 `baseline_matrix`，同语种组不
  产生启用记录（`GET /calibration` 仍为 404）。
- `replaced=true` 表示本次替换了同语种组此前已启用的矩阵。

`GET /calibration?languages=zh&languages=en&languages=ja` 查询同语种组
当前启用矩阵，按查询参数的语种顺序重排行列；无启用记录返回 404。

### 2.5 新会话选用校准矩阵

创建会话时加 `use_calibrated_matrix: true`：

```json
{ "languages": ["ja", "zh", "en"], "use_calibrated_matrix": true }
```

- 服务按语种集合（忽略大小写与顺序）找到同语种组已启用矩阵，**按请求的
  语种顺序重排**后注入 HMM；响应的 `matrix_source` 为 `calibrated`，并
  回显 `transition_matrix`。
- 同语种组尚无启用矩阵时返回 **409 `calibrated_matrix_unavailable`**，
  不创建会话。
- 不传或传 `false` 时沿用 `switch_penalty` 构造的惩罚矩阵，
  `matrix_source` 为 `penalty`。
- 矩阵在**会话创建时快照固定**：之后再校准、替换启用矩阵，已开会话仍用
  原矩阵；只有新会话能拿到新矩阵。校准与收块共用同一把进程锁，校准期间
  块请求短暂排队、不丢块。

### 2.6 错误

业务错误统一返回 `{"error": {"code": "...", "message": "..."}}`：

| HTTP | code | 场景 |
|---|---|---|
| 404 | — | 会话不存在 |
| 409 | `late_chunk` | `start_seq` < 期望序号（迟到/重放旧块但内容不同） |
| 409 | `out_of_order` | `start_seq` > 期望序号（越界、有缺口） |
| 409 | `chunk_conflict` | 同一 `start_seq` 已提交过，但内容或长度不同 |
| 409 | `session_sealed` | 封存后提交新块 |
| 409 | `calibrated_matrix_unavailable` | 建会话时要求校准矩阵，但同语种组尚无启用记录 |
| 404 | `calibration_not_found` | 查询的语种组没有已启用校准矩阵 |
| 413 | `chunk_too_large` | 单块超过 100 000 帧 |
| 422 | `dimension_mismatch` / `negative_probability` / `invalid_distribution` / `non_finite` / `invalid_seq` / `empty_chunk` | 会话块参数或数据校验失败 |
| 422 | `heldout_labels_required` / `label_length_mismatch` / `unknown_label` / `empty_recording` | 校准录音标签问题 |
| 422 | `frame_budget_exceeded` | 校准录音合计超过 1000 帧 |
| 422 | — | Pydantic 层结构校验（如语种数量不在 2～4、录音列表为空） |

**关键不变式**：任何非 2xx 的请求都**不会**改变会话状态（帧序号、区间、
候选切换计数均不变），客户端可以用同一个 `start_seq` 修正后重试。

---

## 3. 实现说明

代码结构：

```
app/smoother.py        核心状态机（纯 NumPy，可独立测试；支持注入校准矩阵）
app/calibration.py     标签约束 EM、期望转移计数、留出条件对数概率、矩阵重排
app/main.py            FastAPI 路由、Pydantic 校验、错误映射、会话表与启用矩阵表
tests/test_api.py      17 个原有用例
tests/test_calibration.py 14 个校准用例
examples/demo.py       分段端到端调用示例
examples/calibration_demo.py 校准端到端调用示例
```

### 3.1 平滑层：带切换惩罚的 HMM 前向滤波

把语种当作隐状态 `z_t ∈ {0..n-1}`，每帧后验 `p_t` 当作观测似然，在对数域
在线做前向滤波：

```
α_t(j) ∝ Σ_i α_{t-1}(i) · A(i,j) · p_t(j)
```

转移矩阵 `A` 给"保持"零对数代价、给"切换到其它语种" `-switch_penalty`，
再按行归一化：

```
log A(i,i) = 0,  log A(i,j) = -penalty (i≠j),  逐行 logsumexp 归一
```

于是 `switch_penalty=0` 时退化为逐帧独立判断；增大惩罚会让状态更不易被单帧
或短时观测拉走。滤波状态 `α` 是**增量**的：每来一帧只做一次 `O(n²)` 更新，
历史不需要重算，块如何切分都不影响结果（见自测
`test_chunk_splits_do_not_change_boundaries`）。

### 3.2 确认层：稳定分段

确认层在平滑状态之上再做一次连续帧门限：

1. **建段**：连续 `switch_frames` 帧被判为同一语种才建立首个区间，起点回退
   到第 0 帧，保证时间线从 0 开始连续。
2. **切段**：当前确认语种之外的候选语种需**连续** `switch_frames` 帧（每帧
   都非低置信）才切换。抖动期帧仍归旧段；真正切换时，旧段在候选起点前一帧
   关闭，新段从候选起点开始——帧不丢、区间首尾相接、边界由"连续达标"的
   位置决定。
3. **低置信**：每帧归一化原始后验的最大值 `< posterior_threshold` 时，该帧
   沿用上一确认语种（并入当前段），并**打断**候选切换的连续计数。
4. **封存**：`seal` 把仍在延伸的尾段按最后一帧关闭（该段此前已通过阈值
   确认），之后会话只读。

已关闭的 `segments` 是追加式列表、元素不可变；重放或后续块返回的都是同一
批对象的内容，因此"已确认区间"对客户端是单调、稳定的。

### 3.3 幂等、顺序与封存

- 每个被接受的块以 `start_seq` 为键缓存：结果快照 + 块长 + 内容指纹
  （形状 + 量化到 1e-9 的字节 SHA-256）。
- 命中缓存且长度、指纹一致 → 返回原结果并置 `duplicate: true`，**不重放**
  状态机，因此不会重复推进 `next_seq`，也不会改变候选切换计数。
- 命中缓存但内容/长度不同 → `409 chunk_conflict`。
- 未命中缓存时，`start_seq` 必须精确等于 `next_seq`：小于即 `late_chunk`，
  大于即 `out_of_order`，两者都不触碰状态机。
- 封存后：未缓存的块返回 `session_sealed`；**已缓存的块仍返回原结果**
  （便于客户端安全重试），响应中 `sealed: true`。

### 3.4 并发

会话表用一把进程内 `threading.Lock` 串行化所有读写，避免同一会话上的块竞争
导致帧顺序错乱。`uvicorn` 以单 worker 运行（`docker-compose.yml` 中显式
`--workers 1`），因为会话状态保存在进程内存。

校准路由与块提交**共用同一把锁**：校准计算期间其它请求短暂排队，既保证
"对照基线是本次校准开始时同语种组的在用矩阵"这一语义稳定，也保证校准期间
到达的块不会丢失（排队后照常处理）。1000 帧、最多 4 语种、5 轮 EM 的计算量
很小（毫秒级），排队影响可忽略。

### 3.5 转移矩阵校准（`app/calibration.py`）

**观测与先验沿用在线 HMM 规则**：每帧后验按行归一化后取对数（下限 1e-12）
作为发射对数似然；每条录音首帧使用均匀先验 `log π(j) = −log n`，与在线
会话首帧一致。

**标签约束的前向-后向**：维护 `(T, n)` 的状态掩码，已标帧只允许人工语种
对应状态，缺标帧全部开放；前向、后向在 `-inf` 掩码下进行。期望转移次数

```
ξ_t(i,j) = α_t(i)·A(i,j)·p_{t+1}(j)·β_{t+1}(j) / P(x)
```

对每条录音的所有相邻帧求和，再跨录音累加；单帧录音不贡献转移计数。
留出打分不使用约束：联合项 `log P(x,z)` 沿人工标注路径累加，证据项
`log P(x)` 由无约束前向得到，二者之差即条件对数概率，天然 ≤ 0。

**EM 更新**：从惩罚 2 矩阵起步，共 5 轮；每轮 `A ← (Σξ + 1) / 行和`
（每种转移一个伪计数，保证任何转移都不为零）。全标语料下一轮即收敛到
"经验转移计数 + 伪计数"的归一化结果，后续轮次保持不变。

**语种组与矩阵重排**：启用矩阵以排序后的小写语种元组为键保存，存储顺序
为校准请求中的语种顺序。会话或查询的语种顺序不同时，按
`M'[i,j] = M[pos(i), pos(j)]` 同时置换行列（可逆，见
`test_new_session_can_use_enabled_matrix_with_reordering`）。

**矩阵快照**：`LanguageSmoother` 在构造时把注入矩阵转成对数矩阵固定下来，
新增 `matrix_source`（`penalty` / `calibrated`）与 `transition_matrix`
查询字段；后续校准替换注册表不影响任何已开会话。默认构造路径与旧版
逐位等价，原有 17 个用例零改动通过。

---

## 4. 验证

```bash
python -m pytest -q
# 31 passed, 1 skipped   （skip 的是需要 SEGMENTER_BASE_URL 的 live 用例）
```

另有 `tests/test_live.py`：设置 `SEGMENTER_BASE_URL` 后对真实运行中的服务发起
HTTP 请求，覆盖健康检查、建会话、正常块、重复块、迟到块、切换 + `final`
封存、封存后拒绝；未设置环境变量时自动跳过。本次交付已在本机用
`uvicorn` 真实启动服务后跑通（`1 passed`）。

### 4.1 校准用例（`tests/test_calibration.py`，14 个）

| 用例 | 验证点 |
|---|---|
| `test_calibration_enables_candidate_and_reuses_baseline_next_time` | 粘性 HMM 合成数据上候选严格胜出并启用；首次基线即惩罚 2 矩阵；第二次校准以上次启用矩阵为对照、`replaced` 语义正确 |
| `test_candidate_not_replaced_when_score_not_higher` | 随机无关数据上候选不胜出时不启用、无启用记录、`active_matrix` 回退基线 |
| `test_first_baseline_is_penalty_two_matrix` | 首次对照矩阵与 `penalty_matrix(n, 2.0)` 逐位相等 |
| `test_train_may_have_missing_labels` | 训练录音整体缺标、逐帧 `null` 均被接受，留出标全 |
| `test_held_out_must_be_fully_labeled` | 留出逐帧缺标/整体缺标 → 422 `heldout_labels_required` |
| `test_frame_budget_and_validation_errors` | 超 1000 帧 422；维度/负概率/零和/未知标签各自 422 且不留启用记录 |
| `test_pydantic_structure_errors` | 语种数越界、空录音列表、空后验等结构错误 422 |
| `test_max_1000_frames_boundary_accepted` | 正好 1000 帧被接受 |
| `test_new_session_can_use_enabled_matrix_with_reordering` | 默认会话仍用惩罚矩阵；`use_calibrated_matrix` 注入启用矩阵；换语种顺序后行列正确重排；GET 查询同步重排 |
| `test_use_calibrated_without_enabled_matrix_is_409` | 同组无启用矩阵 → 409 且不创建会话 |
| `test_existing_session_keeps_its_matrix_after_recalibration` | 再校准替换启用矩阵后，已开会话矩阵不变、新会话取新矩阵 |
| `test_chunks_keep_being_accepted_during_calibration` | 校准进行中并发收块全部成功、序号连续、会话保留建会话时矩阵 |
| `test_smoother_custom_matrix_matches_penalty_construction` | 注入惩罚矩阵与默认构造逐位一致 |
| `test_smoother_rejects_bad_custom_matrix` | 维度错误、行和不为 1、含零的矩阵被核心层拒绝 |
| `test_two_language_calibration_and_404_for_other_group` | 2 语种校准启用；语种组隔离 404；大小写不敏感的同组匹配 |

此外，离线数值核对（不随包交付）用**暴力枚举全部隐状态路径**对照验证了：
带约束前向-后向的期望转移计数与证据值、条件对数概率、全标语料一轮 EM
收敛解、矩阵重排的可逆性，全部在 `1e-10` 容差内一致。

端到端（真实 HTTP 服务）：

```bash
docker compose up --build -d
python examples/demo.py http://127.0.0.1:8000
python examples/calibration_demo.py http://127.0.0.1:8000
SEGMENTER_BASE_URL=http://127.0.0.1:8000 python -m pytest tests/test_live.py -q
docker compose down
```

`demo.py` 输出中的关键片段：

- `chunk 0 (zh, jitter, zh)` → `open_segment` 仍是 `zh 0..7`，`segments` 为空
  （单帧 en 抖动未切段）。
- `duplicate chunk 0` → `duplicate: true`，`next_seq` 仍为 8，`open_segment`
  与首次完全一致。
- `chunk 1 (en x7)` → `segments` 出现 `zh 0..7`，`open_segment` 为 `en 8..14`，
  即切点正好是 en 连续达标的起点。
- `late chunk (start 6)` / `out-of-order chunk (start 20)` → 409，随后
  `next_seq` 仍为 15。
- `chunk 2 (ja x5 + low-confidence x2)` → 切到 ja，两帧低置信并入 ja 段。
- `seal` 后 `segments` 为 `zh 0..7`、`en 8..14`、`ja 15..21`，覆盖全部 22 帧
  且不重叠；封存后新块 409 `session_sealed`。

`calibration_demo.py` 输出中的关键片段：

- `calibration #1`：`baseline_score ≈ -30.5`、`candidate_score ≈ -19.1`，
  `enabled=true, replaced=false, calibration_count=1`，候选矩阵自转移明显
  高于惩罚 2 基线。
- `calibration #2`：同数据再校，对照基线已是上次启用矩阵，分数相等 →
  严格更高才替换，正确返回 `enabled=false` 且计数不变。
- `GET /calibration` 以 `ja, zh, en` 顺序查询，返回按该顺序重排的矩阵。
- 校准矩阵会话 `matrix_source=calibrated` 且照常收块；默认会话仍是
  `penalty`；未校准语种组 `(zh, fr)` 请求矩阵 → 409
  `calibrated_matrix_unavailable`。

> 说明：本次交付环境未安装 Docker，`docker compose up` 未能在本机实际执行；
> Dockerfile / compose 为既有标准写法（本次未改动），服务本身已用
> `uvicorn` 真实启动并跑通两个端到端示例、全部 31 个单测与 1 个 live 用例。

---

## 5. 未完成事项 / 已知限制

1. **无持久化与多实例**：会话与幂等缓存都在进程内存，服务重启即丢失；单
   worker 部署。生产化需要把状态机快照/块结果放到 Redis 或数据库，并按
   `session_id` 做一致性路由或加分布式锁。
2. **无会话过期清理**：长时间运行会累积内存。应加 TTL / LRU 淘汰。
3. **只有前向滤波，没有维特比回溯**：当前段一旦关闭不会因后续帧修正。
   对"整段最优"有要求时，可在封存时对保留的观测做一次离线 Viterbi 重解码
   并重发最终区间（API 已把最终区间集中在 `/segments`，便于替换）。
4. **低置信判定用原始后验而非平滑后验**：阈值语义直观、可解释；若希望阈值
   作用在平滑后的置信度上，需要在 `_apply` 中改为传入过滤后验最大值，并
   补充对应自测。
5. **大块内存占用**：单块上限 100 000 帧，按 `float64` 约 100k×4×8B ≈ 3.2MB
   的请求体；流式超长音频应拆块提交（本就支持）。
6. **没有鉴权、限流与指标**：接口裸奔；接入生产需要 API Key/网关、限流、
   以及 Prometheus 指标（会话数、拒绝原因分布、处理延迟）。
7. **内容指纹量化到 1e-9**：极小概率把"数值上几乎相同"的块判为冲突；如需
   严格语义可改为按块哈希存原始字节。
8. **`seal` 不可撤销**，且封存后不保留重开接口；如需在封存后继续（例如用户
   取消结束），要新增明确的 `resume` 语义。
9. **校准状态同样仅在进程内存**：启用矩阵表 `_calibrations` 随重启丢失，
   且未做持久化审计；生产化应落库并保留每次校准的请求、得分与矩阵版本。
10. **校准在请求内同步计算并持锁**：1000 帧上限下为毫秒级；若未来放宽帧
    预算或语种数，应改为后台任务 + 任务查询接口，避免阻塞块提交。
11. **启用判定只看一次留出分数、无显著性检验**：分数差很小时也可能因严格
    大于而替换；如需更稳健的发布策略，可加交叉验证、最小提升阈值或灰度
    发布（按会话比例选用候选矩阵）。
12. **校准只更新转移矩阵，发射似然仍由识别器后验给定**：若识别器本身存在
    系统性偏差，本接口不会校正；伪计数固定为每种转移 1 个，暂未暴露为参数。
