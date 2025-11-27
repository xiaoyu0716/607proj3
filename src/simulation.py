import os
import sys
import numpy as np
import pandas as pd
import json
import functools
from concurrent.futures import ProcessPoolExecutor

THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from dgps import generate_data
from methods_stable import run_all_methods
from metrics import evaluate

# ---------------------------------------------------------
# TODO: 1. 这里是你要实现的“单次实验”函数
# 把原来循环体里的逻辑搬到这里
# ---------------------------------------------------------
def _run_single_replicate(r, seed0, n, setting, beta_true, alpha_true, beta_false, alpha_false, V_config):
    """
    运行单次模拟实验的辅助函数（为了并行化）。
    必须是顶层函数，不能嵌套在 run_simulation 里面。
    """
    seed = seed0 + r
    
    # 1. 生成数据
    data, truth, beta_used, alpha_used = generate_data(
        n=n,
        setting=setting,
        seed=seed,
        beta=beta_true,
        alpha=alpha_true,
        V_config=V_config,
    )

    # 2. 获取真实值 (从 truth 字典里提取 mu_true 或 ATE_true)
    # ... (你可以参考旧代码) ...
    truth_value = truth.get("mu_true", truth.get("ATE_true"))
    truth_key = "mu_true" if "mu_true" in truth else "ATE_true"
    # 3. 运行所有方法
    # res = ...
    res = run_all_methods(data, setting, beta_true, alpha_true, beta_false, alpha_false)

    # 4. 整理结果并返回
    # return batch_list
    batch = []
    for m, v in res.items():
        batch.append({
            "seed": seed,
            "method": m,
            "estimate": float(v.get("estimate", np.nan)),
            "bias": float(v.get("estimate", np.nan) - truth_value),
            "alpha": v.get("alpha"),
            "beta": v.get("beta"),
            "propensity_spec": v.get("propensity_spec"),
            "outcome_spec": v.get("outcome_spec"),
            "setting": setting,
            "n": n,
            "truth": truth_value,
            "truth_key": truth_key,
        })
    return batch


def run_simulation(setting="A", n=1000, reps=100, seed0=1, beta_true=None, alpha_true=None, beta_false=None, alpha_false=None, n_jobs=-1):
    # Load configuration
    project_root = os.path.dirname(THIS_DIR)
    config_path = os.path.join(project_root, "config", "params.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg_set = cfg["settings"][setting]
    V_config = cfg_set.get("V", None)

    # Resolve parameters (保持旧代码不变)
    if setting == "A":
        if beta_true is None: beta_true = np.array(cfg_set.get("beta_true", [0.0, 1.0, 2.5, 3.0]))
        if alpha_true is None: alpha_true = np.array(cfg_set.get("alpha_true", [-1.0, 1.0, 0.0, 0.0, -1.0]))
        if beta_false is None: beta_false = np.array(cfg_set.get("beta_false", [1.0, 2.5, 3.0]))
        if alpha_false is None: alpha_false = np.array(cfg_set.get("alpha_false", [1.0, -1.0]))
    elif setting == "B":
        if beta_true is None: beta_true = np.array(cfg_set.get("beta_true", [0.0, 2.0, 3.0, 2.0, -4.0]))
        if alpha_true is None: alpha_true = np.array(cfg_set.get("alpha_true", [-3.0, 2.5, 3.0, 1.0, -3.0]))
        if beta_false is None: beta_false = np.array(cfg_set.get("beta_false", [0.0, 2.0, 3.0]))
        if alpha_false is None: alpha_false = np.array(cfg_set.get("alpha_false", [0.0, 2.0]))

    # Prepare results directory
    results_dir = os.path.join(os.path.dirname(THIS_DIR), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # TODO: 2. 这里实现并行调用
    # ---------------------------------------------------------
    print(f"Running simulation for Setting {setting} with n={n}, reps={reps} (Parallel)...")
    
    rows = []
    
    # 使用 functools.partial 固定那些不变的参数
    worker = functools.partial(
        _run_single_replicate,
        seed0=seed0, n=n, setting=setting,
        beta_true=beta_true, alpha_true=alpha_true,
        beta_false=beta_false, alpha_false=alpha_false,
        V_config=V_config
    )
    
    # 启动并行池
    with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        # map 会按顺序返回结果
        results = executor.map(worker, range(reps))
        
        for batch in results:
            rows.extend(batch)
            
    # 保存结果 (和以前一样)
    df_est = pd.DataFrame(rows)
    est_path = os.path.join(results_dir, f"estimates_{setting}.csv")
    df_est.to_csv(est_path, index=False)
    truth_key = df_est["truth_key"].iloc[0]
    truth_value = df_est["truth"].iloc[0]
    # 计算 Metrics (这一步很快，不需要并行)
    # ... (你可以直接把旧代码 calculate metrics 的部分贴过来) ...
    metric_rows = []
    method_names = sorted(df_est["method"].unique())
    for m in method_names:
        vals = df_est.loc[df_est["method"] == m, "estimate"].to_numpy()
        metrics = evaluate(vals, {truth_key: truth_value})  # type: ignore
        metric_rows.append({
            "method": m,
            "setting": setting,
            "n": n,
            "reps": reps,
            "truth": truth_value,
            **metrics,
        })

    df_metrics = pd.DataFrame(metric_rows)
    met_path = os.path.join(results_dir, f"metrics_{setting}.csv")
    df_metrics.to_csv(met_path, index=False)
    
    print(f"Done! Saved to {est_path}")
    return est_path

if __name__ == "__main__":
    # 保护主入口 (Windows/Mac并行计算必需)
    run_simulation(setting="A", n=1000, reps=100, seed0=1)
    run_simulation(setting="B", n=1000, reps=100, seed0=10001)

# if __name__ == "__main__":
#     # --- 临时测试代码 开始 ---
#     print("Testing _run_single_replicate...")
    
#     # 构造一些假的参数，避免 None 报错
#     # 这里的数值不重要，只要格式对就行
#     test_beta_true = np.array([0.0, 1.0, 2.5, 3.0]) 
#     test_alpha_true = np.array([-1.0, 1.0, 0.0, 0.0, -1.0])
#     test_beta_false = np.array([1.0, 2.5, 3.0])
#     test_alpha_false = np.array([1.0, -1.0])
    
#     batch = _run_single_replicate(
#         r=0, seed0=1, n=100, setting="A",
#         beta_true=test_beta_true, alpha_true=test_alpha_true, 
#         beta_false=test_beta_false, alpha_false=test_alpha_false, 
#         V_config=None
#     )
#     # 看看结果长什么样
#     print(f"Result count: {len(batch)}")
#     print(batch[0]) # 打印第一个结果看看
#     print("Test passed!")
#     # --- 临时测试代码 结束 ---

#     # 保护主入口 (Windows/Mac并行计算必需)
#     run_simulation(setting="A", n=1000, reps=100, seed0=1)
#     run_simulation(setting="B", n=1000, reps=100, seed0=10001)