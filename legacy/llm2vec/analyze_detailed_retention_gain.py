import os
import json
import pandas as pd

# 设置路径
BASE_DIR = "./grid_search/results"

# 初始化结果
results = []

# 遍历所有参数组合目录
for run_name in os.listdir(BASE_DIR):
    run_dir = os.path.join(BASE_DIR, run_name)
    if not os.path.isdir(run_dir):
        continue

    path_forget = os.path.join(run_dir, "eval_forget", "evaluation_results.json")
    path_retain = os.path.join(run_dir, "eval_retain", "evaluation_results.json")

    if not os.path.isfile(path_forget) or not os.path.isfile(path_retain):
        continue

    try:
        with open(path_forget, "r") as f:
            forget_data = json.load(f)
        with open(path_retain, "r") as f:
            retain_data = json.load(f)

        diffs = {}
        all_positive = True
        total_diff = 0.0

        for key in retain_data:
            if key in forget_data:
                diff = retain_data[key] - forget_data[key]
                diffs[key] = diff
                total_diff += diff
                if diff <= 0:
                    all_positive = False

        if all_positive:
            record = {
                "param_combo": run_name,
                "total_diff": total_diff,
                **diffs  # 每项差值作为独立列
            }
            results.append(record)
        else:
            record = {
                "param_combo": run_name,
                "total_diff": total_diff,
                **diffs  # 每项差值作为独立列
            }
            if total_diff>0.1:
                print(record)

            print()
    except Exception as e:
        print(f"Error processing {run_name}: {e}")

# 汇总结果
df = pd.DataFrame(results)

# 排序输出
if not df.empty:
    df = df.sort_values(by="total_diff", ascending=False)
    print("✅ 所有 retain-forget 差值均为正的参数组合：\n")
    print(df[["param_combo", "total_diff"]])
    print("\n🌟 差值总和最高的组合：")
    print(df.iloc[0])

    # 保存为 CSV
    df.to_csv("retain_gain_all_positive.csv", index=False)
else:
    print("⚠️ 没有找到所有 retain-forget 差值都为正的组合。")

