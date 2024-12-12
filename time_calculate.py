from typing import List, Dict, Any
import pandas as pd
import os
import json
import time
import numpy as np

# 假设以下函数已在相应的模块中实现
from LDM import initialize_ldm_model_quantization, generate_image
from LLM import initialize_llm_model, generate_story
# metrics相关不再使用，因为我们仅记录时间，不进行指标计算
# from metrics import compute_llm_metrics, compute_story_context_fid, compute_image_metrics

# 固定参数
prompt_type = 'simple'
prompt_order = 'prompt-first'
llm_tokens = 256
image_size = 512
gene_type = 'parallel'

# 输入句子
input_sentence = "Mickey Mouse went to a western restaurant, ordered a roast chicken, and ate it."

# 初始化模型（假设初始化模型是较昂贵的过程，我们在外部初始化一次）
pipe_llm = initialize_llm_model()
single_image_pipeline = initialize_ldm_model_quantization()

# 运行100次inference并记录时间
num_runs = 100
service_times = []

for i in range(num_runs):
    start_pipeline_time = time.time()
    
    # 生成3句故事
    max_attempts = 10
    attempts = 0
    story = []
    while attempts < max_attempts:
        story = generate_story(pipe_llm, input_sentence,
                               type=prompt_type,
                               order=prompt_order,
                               max_tokens=llm_tokens,
                               num_sentences=3)
        if len(story) == 3:
            break
        attempts += 1
    
    # 若在max_attempts内未生成3句故事，跳过本轮（理论上应有错误处理逻辑）
    if attempts == max_attempts:
        print("Failed to generate a 3-sentence story within max attempts.")
        continue
    
    # 根据故事生成图像
    # gene_type = 'parallel' 意味着同时生成3张图像
    if gene_type == 'parallel':
        images = generate_image(single_image_pipeline, prompt=story, size=image_size)
    else:
        # 如果需要 sequential 模式，这里可以写循环，但本次已固定为 parallel
        images = generate_image(single_image_pipeline, prompt=story, size=image_size)

    # 整个流程结束，记录时间
    pipeline_time = time.time() - start_pipeline_time
    service_times.append(pipeline_time)
    print(f"Run {i+1}/{num_runs}: {pipeline_time:.4f} seconds")

# 将结果存为time.csv，一列service_time，共100行
df = pd.DataFrame({'service_time': service_times})
df.to_csv("time.csv", index=False)
print("All results saved to time.csv")
