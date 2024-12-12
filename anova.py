import os
import json
import re
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

output_dir = "output"
data_records = []

def extract_repetition(folder_name: str) -> int:
    # Extract repetition number from folder name
    match = re.search(r"repetition-(\d+)", folder_name)
    if match:
        return int(match.group(1))
    else:
        return None

# Load data
for root, dirs, files in os.walk(output_dir):
    if "result.json" in files and "experiment_config.json" in files:
        result_path = os.path.join(root, "result.json")
        config_path = os.path.join(root, "experiment_config.json")

        with open(result_path, "r") as result_file:
            result_data = json.load(result_file)
        with open(config_path, "r") as config_file:
            config_data = json.load(config_file)

        repetition = extract_repetition(os.path.basename(root))
        
        record = {}
        # Add config parameters
        for k, v in config_data.items():
            record[k] = v

        # Store repetition
        if repetition is not None:
            record["repetition"] = repetition

        # Add result metrics
        for k, v in result_data.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    record[f"{k}_{subk}"] = subv
            else:
                record[k] = v

        data_records.append(record)

df = pd.DataFrame(data_records)

# Convert parameters to categorical as needed
categorical_factors = ["prompt_type", "prompt_order", "gene_type"]
for cat in categorical_factors:
    df[cat] = df[cat].astype("category")

# If you want llm_tokens and image_size to be categorical as well, uncomment:
# df["llm_tokens"] = df["llm_tokens"].astype("category")
# df["image_size"] = df["image_size"].astype("category")

grouping_factors = ["prompt_type", "prompt_order", "llm_tokens", "image_size", "gene_type"]

# Explicitly set observed=True to silence the FutureWarning
df_avg = df.groupby(grouping_factors, as_index=False, observed=True).mean(numeric_only=True)

# Identify numeric columns (these are potential response variables)
numeric_cols = df_avg.select_dtypes(include=[float, int]).columns.tolist()

# Remove grouping factors from numeric_cols if they ended up numeric
for gf in grouping_factors:
    if gf in numeric_cols:
        numeric_cols.remove(gf)

# Also remove repetition if it's in numeric_cols
if "repetition" in numeric_cols:
    numeric_cols.remove("repetition")

# Perform ANOVA for each numeric result variable
for response_var in numeric_cols:
    formula = f"{response_var} ~ C(prompt_type) + C(prompt_order) + C(llm_tokens) + C(image_size) + C(gene_type)"
    model = ols(formula, data=df_avg).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)

    print(f"ANOVA results for {response_var}:")
    print(anova_results)
    print(model.summary())
    print("\n" + "="*80 + "\n")
