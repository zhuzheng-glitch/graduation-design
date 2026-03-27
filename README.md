# Rt-MCMC Project Skeleton

本项目根据开题报告与开题答辩PPT的技术路线搭建，重点对齐“迭代流程”：

- 初始化：EpiEstim 风格初值（窗口 Gamma-Poisson）
- Gibbs 扫描：逐时刻 `R_t` 的 MH 更新
- Gibbs 共轭步：`sigma_rw^2` 的逆伽马更新
- 收敛诊断：多链 `Gelman-Rubin (R-hat)`

## 目录结构

```text
rt_mcmc_project/
  configs/
    simulation.yaml
    covid_case.yaml
  data/
    README.md
  scripts/
    run_simulation.py
    run_covid_case.py
  src/rt_mcmc/
    data/
      io.py
      serial_interval.py
    models/
      renewal.py
      bayesian_rt.py
    mcmc/
      sampler.py
    evaluation/
      metrics.py
    experiments/
      simulation.py
      covid_case_study.py
```

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 运行仿真（平稳/阶跃/指数衰减三类场景）

```bash
python scripts/run_simulation.py --config configs/simulation.yaml
```

输出指标包含：`RMSE`、`90%覆盖率`、`R-hat`、`sigma_rw^2` 后验均值。

3. 运行 COVID 实证（需要准备数据 CSV）

```bash
python scripts/run_covid_case.py --config configs/covid_case.yaml
```

## 数据格式

`data/covid_cases.csv` 至少包含：

- `date`: 日期
- `cases`: 每日新增病例（非负整数）

可选加入 NPI 协变量列（如 `school_close`, `mask_policy`），在配置中写入 `covariate_columns`。
