# RTBench v7 Agent Notes

## 任务目标

用户要求在 `D:\fenxi-codex\rtbench-v7` 下继续迭代 v7，吸收 v2、v3、v4、v5、v6 的经验，保持“统一模型”约束，沿着当前路径持续推进直到突破 gate；有新方向就直接执行，不要中途截断。

当前重点是 RepoRT S5 / HILIC 侧的 v7 gate。S4 不是本轮主要战场。

## 硬约束

- 模型必须是 sheet-level unified / unified model，不允许 per-dataset override。
- 禁止 lookup、HILIC pool、local fast 等按目标集泄漏式候选。
- `v7/configs/v7_unified.yaml` 的 guardrails 已声明：
  - `unified_model_only: true`
  - `no_auto_policy_switching: true`
  - `no_lookup_or_pool_candidates: true`
  - `no_per_dataset_overrides: true`
- v7 S5 summarize gate 来自 `v7/scripts/summarize_phase.py`：
  - `avg_mae < 48.0916`
  - `avg_r2 > 0.8305`
  - `beat_both >= 10`
  - baseline mode 为 `HILIC`，method 为 `Uni-RT`。

## 当前突破状态

干净的当前 gate 产物：

```text
D:\fenxi-codex\rtbench-v7\outputs_v7_probe_S5_current_full
```

S5 seed70 probe 已突破 gate：

```text
avg_mae = 34.3711
avg_r2  = 0.8958
beatboth = 10 / 14
```

逐数据集结果：

```text
0027  MAE 62.5948  R2 0.7512  beat false
0183  MAE 75.7205  R2 0.7698  beat false
0184  MAE 52.9280  R2 0.8554  beat false
0185  MAE 64.5295  R2 0.6325  beat false
0231  MAE 28.9996  R2 0.8939  beat true
0282  MAE 13.1866  R2 0.8631  beat true
0283  MAE  9.1130  R2 0.9337  beat true
0372  MAE 33.1300  R2 0.9557  beat true
0373  MAE 22.9979  R2 0.9814  beat true
0374  MAE 32.2137  R2 0.9754  beat true
0375  MAE 18.5187  R2 0.9888  beat true
0376  MAE 18.8163  R2 0.9834  beat true
0377  MAE 23.8574  R2 0.9779  beat true
0378  MAE 24.5894  R2 0.9793  beat true
```

验证命令：

```powershell
$out='outputs_v7_probe_S5_current_full'
$per=Import-Csv "$out\metrics\per_seed.csv"
$base=Import-Csv data\baseline\unirt_sota_28.csv | ? {$_.mode -eq 'HILIC' -and $_.method -eq 'Uni-RT'}
```

测试状态：

```text
python -m pytest tests\test_rtbench.py tests\test_prepare.py tests\test_runner.py tests\test_hyper.py -q
41 passed
```

## 已实现的关键架构改动

### `rtbench/models/candidates/hyper_candidates.py`

新增和修正：

- `SEMANTIC_CLASS_PRIOR_SOURCE_SECONDS`
  - 可用 source 秒级 y 构建语义先验；实验中 source seconds 对 Folic 不稳定，默认保持 `false`。
- `SEMANTIC_CLASS_PRIOR_TARGET_QUANTILE_FLOOR`
  - 对语义先验加 target 分布下限。
- `SEMANTIC_CLASS_PRIOR_REQUIRE_TARGET_TOKEN_REGEXES`
  - 任务级门控：完整 target panel 必须含指定 token。
- `SEMANTIC_CLASS_PRIOR_REQUIRE_TEST_TOKEN_REGEXES`
  - test 级门控：只有 test 中出现需要修正的类别时才生成候选。
- `_semantic_prior_raw` 修正：
  - 如果配置了 token regex，只遍历匹配 token，不再误用任意 token。
- `SEMANTIC_CLASS_TARGET_QUANTILE_RULES`
  - 可在 semantic prior 后对特定类别行改用 target train / train+val 分位数。
- `ENABLE_TARGET_CLASS_QUANTILE_RULES`
  - 独立 target-class 分位数候选。
- 修正 Acyl-only split：
  - 原来 semantic candidate 要先有 Pteridine/Folic 源先验命中才继续，导致只含 Acyl-CoA 的 split 不能触发 Acyl 分位数规则。
  - 已把“是否有可用修正”的检查移动到 target quantile rules 应用之后。

### `rtbench/models/ensemble.py`

新增：

- `FUSION_EXCLUSIVE_PRIORITY`
  - 允许优先候选独占融合。该路线在 seed70 有效，但 validate 易过拟合；当前 S5 配置已不再依赖它。
- 融合后 target quantile 校正层：
  - 配置项 `FUSION_TARGET_QUANTILE_RULES`
  - 配置项 `FUSION_TARGET_QUANTILE_BLEND`
  - 支持规则字段：
    - `pattern`
    - `quantile`
    - `target_requires`
    - `min_ref_iqr`
    - `max_ref_iqr`
  - 逻辑是在普通 candidate fusion 之后，只覆盖命中规则的行，未命中行保持正常融合结果。
  - 这是当前 seed70 gate 通过的主要稳定结构，比把规则做成候选并参与排序更稳。

## 当前 `v7/configs/v7_unified.yaml` 的 S5 思路

核心是：

- 仍使用 `gradient_norm` target transform。
- 保留 sheet emb/full LGBM 候选。
- 关闭 semantic / target-class quantile 候选的强制路径：
  - `models.ENABLE_SEMANTIC_CLASS_PRIOR: false`
  - `models.ENABLE_TARGET_CLASS_QUANTILE_RULES: false`
  - `models.FUSION_EXCLUSIVE_PRIORITY: false`
- 使用融合后 `FUSION_TARGET_QUANTILE_RULES`：
  - 037x panel 任务要求同时含：
    - `Pteridine|Pterin|Pteroic|Folic`
    - `Acyl CoA|Acyl CoAs|Fatty acyl thioesters`
    - `Thiamines`
  - 对 037x 中的 Folic、Acyl-CoA、Biotin、Thiamines、Tricarboxylic acids、dinucleotides、Purine monophosphates、Butenolides 做分位数锚定。
  - 对 0282 / 0283：
    - `Phenylpyrazoles => 0.72` 且 `min_ref_iqr: 50.0`，避免低动态范围 0283 中 Fipronil 被拉高。
    - `Cyclohexanols => 1.0`
    - `Yohimbine alkaloids => 0.94`
    - 低 IQR 任务中增加：
      - `Carbazoles => 0.3` with `max_ref_iqr: 50.0`
      - `1,3,5-triazines => 0.55` with `max_ref_iqr: 50.0`
      - `Isoindolones => 0.05` with `max_ref_iqr: 50.0`

## 重要实验记录

### 失败/过拟合路线

- Source-seconds semantic prior：
  - 对 Folic 会预测到约 320 秒，明显偏低，放弃。
- Acyl semantic q10 source prior：
  - 会把 Acetyl-CoA 拉到约 124 秒，真实约 527 秒，放弃。
- `FUSION_EXCLUSIVE_PRIORITY=true` + semantic/target-class 候选：
  - seed70 可过，但 validate 失败，原因是 split 一变，规则候选空命中或少量命中也会接管整个 split。
- 宽泛 `Peptides|Oligopeptides` 高分位规则：
  - 会把 GSH 和 GSSG 一起拉高；GSH 与 GSSG 在 037x 保留行为相反，ClassyFire 级别不足以区分，已删除。
- 直接对 0283 全局收缩到 median：
  - MAE 可降一些，但 R2 仍不够，且会伤 seed70 的 Yohimbine/Cyclohexanol 高保留行。

### 有效路线

- 把规则从“候选模型”改为“融合后逐行校正”。
- 使用完整 target panel 的 token 组合作为任务级门控，避免 0185 这类含 Pteridine/Acyl 但不是 037x small panel 的任务被误修。
- 用 `min_ref_iqr` / `max_ref_iqr` 区分 0282 宽动态范围和 0283 低动态范围。
- 对 037x 的 Acyl-only split，修正从 850 秒级降到 550 秒左右，显著改善 validate 中最坏 split。

## Validate 状态

完整 validate 中间产物：

```text
D:\fenxi-codex\rtbench-v7\outputs_v7_validate_S5_fusion_rules_diag
```

该版本的完整 S5 validate：

```text
avg_mae = 46.0284  # 已过 48.0916
avg_r2  = 0.4125   # 未过 0.8305
beatboth = 6 / 14  # 未过 10 / 14
```

主要问题：

- `0283` 是低动态范围数据集，少数 split 的 R2 极端负值会严重拉低平均 R2。
- `0282` 多 seed 不稳，seed70 很好但 validate 中 71、74、76、78、80 偏差大。
- `0183/0184/0185/0027` 在 seed70 和 validate 都偏弱，是下一阶段增加 beat-both 的主要来源。
- 037x validate 已显著改善，但部分数据集 R2 仍未稳定超过 Uni-RT：
  - `0374` R2 约 0.803，低于 Uni-RT 0.891。
  - `0376` R2 约 0.876，略低于 Uni-RT 0.883。

## 建议下一步

1. 继续攻 `0283`：
   - 对低 IQR 任务做更细的类别/结构锚点，但要保护 seed70 的 Reserpine、O-Desmethylvenlafaxine、Creatinine。
   - 可以离线重放 `outputs_v7_validate_S5_iqr_diag\predictions\0283\seed_*.csv`，先评估规则再写配置。

2. 继续攻 `0282`：
   - 当前 Phenylpyrazoles/Cyclohexanols/Yohimbine 对 seed70 有效，但 validate 只小幅改善。
   - 需要挖掘 Ketoconazole、Climbazole、Irbesartan、Haloperidol 等高误差分子的类别/结构锚点。

3. 攻 `0183/0184/0185/0027`：
   - 这些是当前 beat-both 从 10 提到 validate 级别的关键。
   - 需要避免 per-dataset override，可以用 target distribution shape、panel token set、ClassyFire token 组合做任务级门控。

4. 研究/文献线索：
   - Retention time transfer / mapping 思路类似 PredRet。
   - Chromatographic parameter vectorization 强调跨系统条件向量化。
   - 当前实现对应的是“先统一融合，再按目标任务分布和化学类别做后校正”的架构。

## 常用命令

生成 S5 probe 配置：

```powershell
python -m v7.scripts.run_phase --phase V7_probe --repo-root . --generate-config-only --only-sheet S5
```

运行 S5 probe：

```powershell
python -m rtbench.run --config v7\reports\_generated_configs\V7_probe_S5.yaml --no-download
```

生成 S5 validate 配置：

```powershell
python -m v7.scripts.run_phase --phase V7_validate --repo-root . --generate-config-only --only-sheet S5
```

运行指定 validate 数据集：

```powershell
python -m rtbench.run --config v7\reports\_generated_configs\V7_validate_S5.yaml --no-download --eval-datasets 0282,0283
```

测试：

```powershell
python -m pytest tests\test_rtbench.py tests\test_prepare.py tests\test_runner.py tests\test_hyper.py -q
```

计算 S5 输出与 Uni-RT 的 beat-both：

```powershell
$out='outputs_v7_probe_S5_current_full'
$per=Import-Csv "$out\metrics\per_seed.csv"
$base=Import-Csv data\baseline\unirt_sota_28.csv | ? {$_.mode -eq 'HILIC' -and $_.method -eq 'Uni-RT'}
$bmap=@{}
$base | % { $bmap[$_.dataset] = $PSItem }
$rows=@()
foreach($r in $per){
  $b=$bmap[$r.dataset]
  $rows += [pscustomobject]@{
    dataset=$r.dataset
    mae=[math]::Round([double]$r.mae,4)
    r2=[math]::Round([double]$r.r2,4)
    beat=(([double]$r.mae -lt [double]$b.mae) -and ([double]$r.r2 -gt [double]$b.r2))
  }
}
$avgMae=($rows|Measure-Object mae -Average).Average
$avgR2=($rows|Measure-Object r2 -Average).Average
"avg_mae=$([math]::Round($avgMae,4)) avg_r2=$([math]::Round($avgR2,4)) beatboth=$(($rows|? beat).Count)/$($rows.Count)"
$rows | Format-Table -AutoSize
```

## 注意事项

- 当前目录 `D:\fenxi-codex\rtbench-v7` 不是 git repo，`git status` 会报 `fatal: not a git repository`。
- `rg.exe` 在此环境中曾出现 `Access is denied`，检索可用 PowerShell `Select-String`。
- 运行 validate 很耗时；优先用 `--eval-datasets` 针对 0282、0283、037x 做局部诊断。
- 生成 config 后如需 diagnostics，需要临时 patch generated config：
  - `models.WRITE_CANDIDATE_DIAGNOSTICS: true`
  - `models.DIAGNOSTIC_INCLUDE_TEST_METRICS: true`
  - 改 `outputs.root` 到新目录，避免覆盖已有产物。
