# CTS（Concurrent Teacher-Student）移植计划

> 将 `~/LeggedGym-Ex` 中的 `ppo_cts` 移植到 `~/rsl_rl`（当前分支：`dev_cts`）
>
> 状态：**方案已确定（v2 含零浪费设计），待实施**（P0 起）

## 0. 背景与探索结论

### 0.1 CTS 是什么

**CTS = Concurrent Teacher-Student（并发师生蒸馏）**，参考 [Concurrent TS](https://clearlab-sustech.github.io/concurrentTS/)。**不是** curiosity / count-based 探索。

核心机制：
- teacher 与 student **共享同一个 actor MLP**，区别仅在 latent 来源：
  - teacher：`latent = privilege_encoder(privileged_obs)`
  - student：`latent = history_encoder(obs_history)`
  - 两者最终：`action_mean = actor(cat(obs, latent))`
- 训练时前 `num_teacher` 个 env 为 teacher，其余为 student，**并发训练**

### 0.2 源端损失结构（已逐行确认，`ppo_cts.py:_compute_rl_loss` L193-278）

```python
loss = value_loss_coef * value_loss
     + teacher_surrogate_loss + student_surrogate_loss   # 双 surrogate，各自 .mean()
     - entropy_coef * cat(teacher_entropy, student_entropy).mean()
```

- teacher/student 的 advantage **各自独立归一化**
- KL 自适应学习率**只用 teacher**（teacher 的 mu/sigma）
- 额外损失：`reconstruction_loss = MSE(history_encoder(obs_history), privilege_encoder(privileged_obs).detach())`（仅学生样本）
- 两个优化器：RL 参数（actor + critic + privilege_encoder + std）与 history_encoder 独立

### 0.3 源端集成面（LeggedGym-Ex）

| 文件 | 角色 |
|---|---|
| `algorithms/ppo_cts.py` | `PPO_CTS(PPO)` 算法（278 行） |
| `modules/actor_critic_cts.py` | `ActorCriticCTS`（独立 nn.Module，非 ActorCritic 子类） |
| `storage/rollout_storage_cts.py` | `RolloutStorageCTS`（分组 advantage + 17 元组 mini-batch） |
| `runners/cts_runner.py` | `CTSRunner`（4 观测流 + 4 返回值 update） |
| `legged_gym/envs/base/legged_robot_{cts,ts}.py` | env 侧：obs_history 堆叠、critic_obs、num_teacher 划分 |
| `utils/symmetry_cts.py` / `ppo_cts_amp.py` / `cts_amp_runner.py` | 可选组合（对称增强 / AMP），机器人特定，**本次不移植** |

### 0.4 目标端架构（rsl_rl，已逐项核实）

| 方面 | 目标端现状 |
|---|---|
| obs | 完整 `TensorDict`，按 `obs_groups` 分组取用 |
| actor/critic | `MLPModel` 独立模块（`models/mlp_model.py`），接口：`forward/get_output_log_prob/output_distribution_params/output_entropy/get_hidden_state/reset/update_normalization/get_kl_divergence/is_recurrent` |
| 算法构造 | `construct_algorithm`（静态方法，ppo.py L686-746）：`resolve_class(cfg["algorithm"])`、`resolve_class(cfg["actor"])`、`resolve_obs_groups`、`RolloutStorage("rl", ...)` |
| class_name 解析 | `resolve_callable`（utils.py L98+）：支持 `"module.path:Class"` 全限定名 → **新增类无需改 `__init__.py`** |
| storage | `self.observations = TensorDict({key: zeros... for key in obs.items()})`（L142-149）→ **保留 obs 全部 key 且全量预分配**（含显存影响，见 §2.2）；mini-batch 随机采样（env 身份丢失） |
| 扩展钩子 | 已有 15 个 AMP 钩子：`_process_step_rewards` / `_extra_mini_batch_iter` / `_compute_aux_loss` / `_compute_aux_gradients` / `_step_aux_optimizers` / `_train_aux_modules` / `_eval_aux_modules` / `_compile_aux_modules` / `_aux_save_state` / `_load_aux_state` / `_aux_parameters` / `_aux_broadcast_state_dicts` / `_load_aux_broadcast_state_dicts` |
| update() | actor 前向（L408-416）与 surrogate（L450-456）**内联**（无可覆写方法）；返回 loss dict |
| runner | `act(obs)` / `compute_returns(obs)` / `update()` 接口统一；`get_inference_policy()` 走 `alg.get_policy()`（含 as_jit/as_onnx 导出） |
| logger | 按 `update()` 返回的 loss_dict 通用记录指标 |

### 0.5 关键架构差异 → 不能 drop-in 移植

源端（raw tensor、单 ActorCritic、分组存储、子类化）与目标端（TensorDict、MLPModel、钩子化、dict 返回）差异大，需按算法逻辑重写集成。

## 1. 移植原则

| # | 原则 | 落实方式 |
|---|---|---|
| ① | 以当前 PPO 实现优先，只参考 CTS 逻辑/idea | 只移植算法逻辑；`PPO_CTS` 子类复用当前 PPO 全部机制 |
| ② | 对本项目代码修改控制在最小范围 | 基类仅 4 处**手术式行为保持**改动（见 §3.1），其余全部为新增文件 |
| ③ | CTS 作为拓展/插件，cfg 开关选择 | `cts_cfg` + `class_name` 切换，不启用时零影响；mjlab 侧 `RslRlCtsCfg` dataclass |
| ④ | 为 ppo_ee / ppo_dreamwaq 提前考量 | `_compute_policy_loss` / `_normalize_advantages` / aux 指标三处钩子为三者公共扩展点 |

## 2. 总体设计（推荐方案：子类 + 手术式钩子 + 零浪费存储）

```
mjlab cfg: class_name="PPO_CTS", actor.class_name="CtsActor", cts_cfg={...}, obs_groups={...}
    │
    ▼ (runner 零改动，resolve_callable 解析)
rsl_rl/algorithms/ppo_cts.py:  PPO_CTS(PPO)
    ├─ _compute_policy_loss()      ← 按 teacher_mask 拆分组，双前向 + 双 surrogate【新增钩子】
    ├─ _normalize_advantages()     ← teacher/student 区域各自归一化【新增钩子】
    ├─ _compute_aux_loss()         ← reconstruction loss（复用现有 aux 钩子，仅 student 样本）
    ├─ _compute_aux_gradients()/_step_aux_optimizers()  ← history_encoder 优化器
    └─ _aux_save_state()/_load_aux_state()              ← 编码器优化器状态
    │
    ▼
rsl_rl/storage/rollout_storage_cts.py:  RolloutStorageCTS(RolloutStorage)
    ├─ 主存储剥离 "history"（policy/privileged/critic/teacher_mask 等廉价组，全 env）
    ├─ observation_histories: [T, num_students, history_dim]  ← student-only，零 teacher 浪费
    └─ mini_batch_generator 附带 batch.indices（样本→(t, env) 映射，供取 history）
    │
    ▼
rsl_rl/models/actor_critic_cts.py:  CtsActor(nn.Module)  ← 实现 MLPModel 同一接口
    ├─ privilege_encoder (MLP) + history_encoder (MLP) + actor MLP(cat(obs, latent))
    ├─ forward(obs) 索引拆分：teacher 行 privilege_encoder / student 行 history_encoder（互不读取）
    └─ get_policy()/as_jit()/as_onnx() 部署模式 = student（全部用 history_encoder）
```

### 2.1 关键机制（已验证可行）

1. **teacher_mask 随 obs 流转**：env 在 obs TensorDict 提供 `teacher_mask`（前 `num_teacher` 个 env 为 1）。它属于廉价组，进主存储；mini-batch 采样后每个样本仍可识别 teacher/student。
2. **区域化 advantage**：`compute_returns` 的 GAE 递归不变（teacher/student 共享 critic），仅把最终归一化改为 `envs[:num_teacher]` 与 `envs[num_teacher:]` 各自归一化，写入同一张 `advantages` 张量。**零新增缓冲**。
3. **student-only history 缓冲（零浪费核心）**：`"history"` **不进主存储**，由 `RolloutStorageCTS` 单独分配 `[T, num_students, history_dim]`；`add_transition` 从整包 obs 中切出 `[num_teacher:]` 存入、再以 `TensorDict.exclude("history")`（廉价视图）走基类逻辑。→ **teacher 的持久占用 = 0**。
4. **按组前向 + batch.indices**：mini-batch 随机采样后，`Batch` 携带 `indices`（扁平索引 → 可反推 t、env）。`_compute_policy_loss` 按 teacher_mask 拆分：teacher 子集前向**无 history**（零瞬时浪费），student 子集前向从 student 缓冲按 indices 取 history。→ **teacher 的瞬时占用 = 0**。

### 2.2 显存账（以当前 history_dim = frame_stack×num_observations = 900 fp32、T=24、4096 env = 3072 teacher + 1024 student 为例）

| 项 | v1（history 进主存储） | v2（零浪费） |
|---|---|---|
| 持久 teacher history `[T, 3072, 900]` | **265 MB 白占** | **0** |
| 持久 student history `[T, 1024, 900]` | 88 MB | 88 MB（不可避免） |
| 更新期瞬时 history（每批） | 88 MB（含 66 MB teacher 零） | 22 MB（仅 student 子集） |
| env 每步 obs 缓冲 `[4096, 900]` | 15 MB（含 teacher 11 MB） | 15 MB（**唯一残余**，standing 不随 T 累加） |

> 当前 student 使用的 history 为**堆叠观测**（`frame_stack×num_observations`，如 20×45=900）。未来加入**深度图模态**时 history_dim 会大幅增大（如 96×96=9216 时上表各项 ×10，teacher 持久浪费达 ~2.5 GiB）——这正是本方案采用零浪费设计的前瞻动机。
>
> **唯一无法消除的残余**：env 每步返回的 obs TensorDict 必须含 `"history"`（`[4096, history_dim]`，teacher 行填零）——actor 前向要读、TensorDict batch 维必须一致。它是一个**复用型 standing 缓冲**（当前 ~15MB），不随 T 累加。要连这个也消掉需改 runner→act 传参接口（history 走独立通道），改动面大，暂不接受。

## 3. 改动清单

### 3.1 基类手术式改动（4 处，全部行为保持）

| # | 文件 / 位置 | 改动 | 默认行为 |
|---|---|---|---|
| 1 | `algorithms/ppo.py` `update()` L408-456 | 把"actor 前向 + log_prob + distribution_params/entropy 切片 + KL/LR + surrogate"整块提取为 `_compute_policy_loss(batch, original_batch_size) -> (surrogate_loss, entropy)` | 与现行内联代码完全一致 |
| 2 | `algorithms/ppo.py` `compute_returns()` L213-214 | 提取 `_normalize_advantages()` 钩子 | 现行全局归一化 |
| 3 | `algorithms/ppo.py` `update()` L363-370 / 517-520 / 538-541 / 553-556 | AMP 专用指标累加泛化为 "`_compute_aux_loss` 返回的 metrics dict 通用累加进 loss_dict" | AMP 指标键不变，行为等价 |
| 4 | `storage/rollout_storage.py` `Batch`（L24-65） | 增加 `indices: torch.Tensor \| None = None` 字段；`mini_batch_generator`（L225-258）填充 | 现有字段不变，向后兼容 |

- 第 1 处是 v2 的关键改动：critic 前向、value loss、RND/symmetry/AMP、backward、step **全部留在基类**（只依赖 batch 廉价组，history 不进 batch 无影响）。
- 第 3 处为小重构；AMP 的 `amp/score_agent` 等键名保持不变，由现有测试 + AMP smoke 回归。
- 为 EE/DreamWaQ 预留：三者都需要自定义策略前向（EE 拼 estimator 输出、DreamWaQ 拼 z/vel）与额外 loss 指标。

### 3.2 新增文件

**A. `rsl_rl/storage/rollout_storage_cts.py` — `RolloutStorageCTS(RolloutStorage)`**（~90 行）
- `__init__`：剥离 `"history"` 后调 `super().__init__()`（主存储无 history）；另建 `self.observation_histories = zeros(T, num_students, *history_shape)`
- `add_transition`：`observation_histories[self.step] = transition.observations["history"][num_teacher:]`，再 `transition.observations = transition.observations.exclude("history")` 后走基类
- `mini_batch_generator`：采样逻辑同基类，`Batch.indices` 填充扁平索引（可反推 `t = idx // num_envs`、`env = idx % num_envs`）
- 辅助方法：`get_student_history(batch_indices) -> Tensor`（按 `t*num_students + (env - num_teacher)` 映射取 student 历史）

**B. `rsl_rl/models/actor_critic_cts.py` — `CtsActor(nn.Module)`**（~180 行）
- 内部构建：`privilege_encoder`（MLP）、`history_encoder`（MLP）、actor MLP（输入 = `policy_obs_dim + latent_dim`）、distribution 头
- 复用现有构建块：`get_activation`、`modules/mlp.py:MLP`、`modules/distribution.py` 分布类
- 实现 MLPModel 同一接口（mlp_model.py L82-189）：`forward(obs, masks, hidden_state, stochastic_output)`、`get_output_log_prob`、`output_distribution_params`、`output_entropy`、`get_hidden_state`、`reset`、`update_normalization`（no-op）、`is_recurrent=False`、`get_kl_divergence`
- `forward` **索引拆分**（非掩码全算）：`latent[teacher_idx] = privilege_encoder(obs["privileged"][teacher_idx])`、`latent[student_idx] = history_encoder(obs["history"][student_idx])`——teacher 行不读 history（不浪费计算、不碰零值），拼接 policy obs 后过 actor MLP
- 导出封装 `as_jit()/as_onnx()/get_dummy_inputs()`：部署 forward = **student 模式**（全部走 history_encoder），对应源端 `act_student`
- `get_kl_divergence`：默认整批 KL（见决策 D4）

**C. `rsl_rl/algorithms/ppo_cts.py` — `PPO_CTS(PPO)`**（~170 行）
- `__init__`：`super().__init__(...)` 后从 `self.actor.history_encoder` 建 `encoder_optimizer`（lr 取 `cts_cfg["encoder_lr"]`）
- `_compute_policy_loss(batch, original_batch_size)`：
  - `mask = batch.observations["teacher_mask"][:original_batch_size]`；teacher 子集 / student 子集按 indices 切片
  - teacher 前向：obs 无 history → `self.actor(teacher_obs, ...)` → teacher surrogate（公式对齐源端 L226-231）
  - student 前向：obs + `storage.get_student_history(batch.indices)` 组 TensorDict → student surrogate（对齐源端 L240-246）
  - KL/LR：teacher-only（源端）或合并（D4 决策）
  - 返回 `(teacher_surrogate + student_surrogate, cat(teacher_entropy, student_entropy))`
- `_normalize_advantages`：teacher 区（`st.advantages[:, :num_teacher]`）与 student 区各自 `(x - x.mean())/(x.std()+1e-8)`；`num_teacher` 存实例
- `_compute_aux_loss(batch, disc_obs_batch)`：`MSE(history_encoder(student_history), privilege_encoder(batch["privileged"][student]).detach())`，返回 `(loss, {"cts/reconstruction": loss})`
- `_compute_aux_gradients` / `_step_aux_optimizers`：encoder_optimizer（**显式 zero_grad**，见风险 5）
- `_train_aux_modules` / `_eval_aux_modules`：编码器 train/eval（actor 本体由基类管）
- `_aux_save_state` / `_load_aux_state`：encoder_optimizer state_dict（actor 内编码器参数随 `_raw_actor` 自动保存，L751 已核对）
- `get_policy()`：返回 CtsActor（含导出封装）

**D. 可选 `rsl_rl/extensions/cts.py` — `resolve_cts_config()`**
- 若想让 `cts_cfg` 自动推断 obs group 维度（对齐 `resolve_amp_config` 模式）则加；若 mjlab 显式提供 obs_groups 映射则可省略

### 3.3 注册（零 `__init__.py` 改动）

利用 `resolve_callable`（utils.py L98+，支持 `"module.path:Class"` 全限定名），cfg 直接写：

```python
"algorithm": {
    "class_name": "rsl_rl.algorithms.ppo_cts:PPO_CTS",
    "cts_cfg": {...},
},
"actor": {
    "class_name": "rsl_rl.models.actor_critic_cts:CtsActor",
    "num_teacher": N,                     # 经 cfg 传入（env 侧需一致）
    "privilege_encoder_hidden_dims": [256, 128],
    "history_encoder_hidden_dims": [256, 128],
    ...
},
```

基类 `construct_algorithm`（L686-746）已核对：actor 由 `cfg["actor"]` 的 class_name 解析构建（L712）→ **actor 构造零改动**。
storage 构造处（L728 `RolloutStorage("rl", ...)`）由 `PPO_CTS.construct_algorithm` 覆写为 `RolloutStorageCTS`（唯一覆写点，其余流程复用基类）。

### 3.4 mjlab 侧（另一仓库，不在本仓库改动范围）

- `src/mjlab/rl/config.py`：新增 `RslRlCtsCfg` dataclass + `RslRlPpoAlgorithmCfg.cts_cfg: RslRlCtsCfg | None = None`
- env：obs TensorDict 提供 `"policy" / "privileged" / "history" / "critic" / "teacher_mask"`；`obs_history` 堆叠（frame_stack，参考源端 `legged_robot_ts.py`）；`num_teacher` 属性
- **env 每步返回的 `"history"` 为 `[4096, history_dim]`，teacher 行填零**（standing 缓冲，见 §2.2 残余说明）
- 任务 cfg：`class_name` 指向 PPO_CTS / CtsActor；`obs_groups` 映射 `{"actor": ["policy"], "critic": ["critic"], "privileged": ["privileged"], "history": ["history"]}`
- **num_teacher 语义与源端一致：前 `num_teacher` 个 env 是 teacher**（源端 go2 配置 3072/4096，teacher 占多数）

### 3.5 明确零改动清单

- `runners/on_policy_runner.py`（act / compute_returns / update 接口相同，class_name 解析已支持）
- `utils/logger.py`（reconstruction / surrogate 指标随 `update()` 返回的 loss_dict 通用记录）
- `extensions/amp.py`、`storage/circular_buffer.py`（CTS 不需要）
- `storage/rollout_storage.py` 仅第 3.1-#4 一处字段级增加（`Batch.indices`）

## 4. 设计决策（含备选与推荐理由）

| 决策 | 推荐 | 备选 | 理由 |
|---|---|---|---|
| D1 集成机制 | **PPO_CTS 子类 + 4 处手术式改动** | 纯 add-on（整体覆写 update/compute_returns，零基类改动但重复 ~200 行）；全钩子融入基类 PPO（最"插件"，但需大改 act/update 核心，违反原则②） | 复用当前 PPO 最大化（原则①）、基类改动最小且行为保持（原则②）、钩子为 EE/DreamWaQ 共用（原则④） |
| D2 history 存储 | **student-only 缓冲 + 按组前向（零浪费）** | v1 合并采样 + teacher_mask（history 进主存储，`[T, 3072, history_dim]` 白占）；源端 17 元组分批存储 | 彻底消除 teacher 持久/瞬时占用（当前 history 即可省 ~265 MB@T=24；未来 depth 模态下 history_dim 增大数倍时收益更显著）；且按组前向与源端 `_compute_rl_loss` 结构更接近（更忠实） |
| D3 actor 构造 | **经 `cfg["actor"]` class_name 自动构建** | `_create_actor` 工厂钩子 / 覆写 construct_algorithm | 基类 construct_algorithm 已支持，仅 storage 构造一处覆写 |
| D4 自适应 KL | **teacher-only KL**（源端行为，默认） | 整批 KL：CtsActor forward 时暂存 mask，`get_kl_divergence` 应用 | v2 已按组前向，teacher-only KL 与源端一致且实现成本低；如需整批可低成本切换 |
| D5 CTS+AMP 组合 | **暂不做**，架构天然支持 | PPO_CTS 覆写 `_compute_aux_loss` 时 `super()` 链 AMP 损失再叠加重建损失 | 目标端 AMP 是钩子形式，组合只需链式调用 |

## 5. 分阶段实施

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| P0 | 确认 `dev_cts` 分支基线（c7758d4），跑通现有测试 | 基线绿 |
| P1 | 基类 4 处手术式改动（含 `_compute_policy_loss` 提取） | 全测试 + AMP smoke 回归通过，diff 仅 ~70 行 |
| P2 | `RolloutStorageCTS` + `CtsActor` + `PPO_CTS` | 单元级：接口对齐 MLPModel / Batch；`get_student_history` 索引映射正确（mock env 单测 forward/export/存储切片） |
| P3 | mjlab 侧：`RslRlCtsCfg` + env obs groups/teacher_mask + 任务 cfg | 配置解析端到端（asdict → construct_algorithm） |
| P4 | 端到端 smoke：小规模（如 64 env × 50 iter）CTS 训练 | 训练步进、loss dict 含 cts/reconstruction、teacher/student surrogate 收敛迹象；**显存核算**（teacher history 占用为 0） |
| P5 | 多卡广播/规约 + 保存/加载 + ONNX 导出（student 模式） | 无 NaNs、checkpoint 回放一致、导出成功 |
| P6 | README 扩展清单更新 | 文档 |

## 6. 风险与开放问题

1. **环境侧工作量最大**：obs_history 堆叠、critic_obs（c_frame_stack）、teacher_mask、domain randomization 对齐——全在 mjlab env 侧，且任务相关（需与现有 AMP 观测组共存）。
2. **忠实度已对齐**：按组前向 + 双 surrogate 与源端 `_compute_rl_loss` 结构一致；仅 `_normalize_advantages` 的 per-mini-batch 交互与 D4 的 KL 选择需按默认配置验证。
3. **`_normalize_advantages` 与 `normalize_advantage_per_mini_batch` 的交互**：per-mini-batch 模式下区域归一化发生在 batch 内（基类 L396-398 已有整批逻辑，CTS 覆写钩子时需同步考虑 mask 拆分）。默认 `False` 不受影响。
4. **`_compute_policy_loss` 提取的回归面**：该块含 KL/LR 自适应与多卡 LR 广播（L423-448），提取后需确保 AMP/RND/Symmetry 各组合下行为不变（P1 全量回归覆盖）。
5. **encoder_optimizer 梯度清零**：基类 `update()` 的 `self.optimizer.zero_grad()`（L482）不覆盖 history_encoder（不在 RL 优化器内），`_compute_aux_gradients` 必须显式 `encoder_optimizer.zero_grad()`，否则重建梯度跨 mini-batch 累积（CTS-only 与 CTS+AMP 均适用）。
6. **env standing 缓冲残余**：`[4096, history_dim]` 每步缓冲（teacher 行填零）为已知不可消残余（§2.2），如需消除须改 runner→act 接口，超出本次范围。
7. **encoder_optimizer 调度**：源端 history_encoder 用固定 lr（不参与自适应调度），计划保持。

## 附录 A：源端参考文件

| 文件 | 用途 |
|---|---|
| `~/LeggedGym-Ex/rsl_rl/algorithms/ppo_cts.py` | 算法逻辑（L193-278 损失结构、L226-246 双 surrogate 公式） |
| `~/LeggedGym-Ex/rsl_rl/modules/actor_critic_cts.py` | 编码器 + actor 结构 |
| `~/LeggedGym-Ex/rsl_rl/storage/rollout_storage_cts.py` | 分组 advantage / 分组 mini-batch 语义（参考语义，结构按目标端重写） |
| `~/LeggedGym-Ex/rsl_rl/runners/cts_runner.py` | env 观测契约 |
| `~/LeggedGym-Ex/legged_gym/envs/base/legged_robot_ts.py` | env 侧 obs_history / critic_obs 构造 |

## 附录 B：目标端关键接口（实现时对照）

| 接口 | 位置 |
|---|---|
| `PPO.construct_algorithm`（静态） | `rsl_rl/algorithms/ppo.py` L686-746 |
| `PPO.act` | L144-155 |
| `PPO.process_env_step` | L157-187 |
| `PPO.compute_returns` | L189-214 |
| `PPO.update` | L353-561 |
| `PPO._compute_policy_loss` 提取块（actor 前向+KL+surrogate） | L408-456 |
| `PPO._compute_aux_loss` 等 15 个钩子 | L216-351 |
| `RolloutStorage.Batch`（`indices` 新增字段） | `rsl_rl/storage/rollout_storage.py` L24-65 |
| `RolloutStorage.__init__`（全量存 obs，v2 由 CTS 子类剥离 history） | L125-172 |
| `RolloutStorage.mini_batch_generator` | L225-258 |
| `MLPModel` 接口 | `rsl_rl/models/mlp_model.py` L82-189 |
| `resolve_callable` / `resolve_class` | `rsl_rl/utils/utils.py` L98 / L178 |
| `resolve_obs_groups` | `rsl_rl/utils/utils.py` L191-286 |
| runner inference/export | `rsl_rl/runners/on_policy_runner.py` L168-206 |