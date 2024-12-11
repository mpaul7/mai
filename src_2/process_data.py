import pandas as pd

train_file="/home/mpaul/projects/mpaul/mai/data/train_dalhousie_nims_7app_nfs.parquet"
test_file="/home/mpaul/projects/mpaul/mai/data/test_dalhousie_nims_7app_nfs.parquet"

df = pd.read_parquet(train_file)
print(df.columns)
# Copy sequence packet columns with _cnn suffix
df['splt_direction_cnn'] = df['splt_direction']
df['splt_ps_cnn'] = df['splt_ps'] 
df['splt_piat_cnn'] = df['splt_piat']
