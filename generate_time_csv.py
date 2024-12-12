import pandas as pd
import numpy as np

# 假设已有的 time.csv 文件中有一列 service_time，共100行
df = pd.read_csv("time.csv")
assert 'service_time' in df.columns, "time.csv 中应有service_time列"
assert len(df) == 100, "time.csv 行数应为100，请检查数据。"

# 定义四种场景的到达率
arrival_rates = {
    'low': 1/30,          # 约0.0333
    'near': 1/25,         # 0.04
    'high': 1/10,         # 0.1
    'ultra': 3/10,        # 0.3
}

# 函数：给定lambda生成M和G两种inter_arrival_time
def generate_inter_arrival_times(n_samples, lam):
    # M: 指数分布
    inter_arrival_M = np.random.exponential(scale=1/lam, size=n_samples)

    # G: Gamma分布，以shape=2为例
    shape = 2.0
    # 使Gamma分布均值约为1/lam => mean=shape*scale => scale=(1/lam)/shape
    scale = (1/lam)/shape
    inter_arrival_G = np.random.gamma(shape=shape, scale=scale, size=n_samples)
    
    return inter_arrival_M, inter_arrival_G

# 为每种场景生成M和G列
for scenario, lam in arrival_rates.items():
    inter_arrival_M, inter_arrival_G = generate_inter_arrival_times(100, lam)
    df[f'inter_arrival_time_M_{scenario}'] = inter_arrival_M
    df[f'inter_arrival_time_G_{scenario}'] = inter_arrival_G

# 保存为新的CSV文件
df.to_csv("time.csv", index=False)
print("Updated time.csv with 9 columns successfully created.")
