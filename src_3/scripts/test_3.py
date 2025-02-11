import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# df = pd.read_parquet('/home/mpaul/projects/enta_dl/enta_workspace/data/enta_data/enta_solana2022a_prepared_data_nfs/feature_view_prepared_data/train_test_split/test_enta_Solana2022a_6c_nfs_allfeat.parquet')
# df = pd.read_parquet('/home/mpaul/projects/enta_dl/enta_workspace/data/enta_data/enta_solana2022a_prepared_data_nfs/feature_view_prepared_data/train_test_split/train_enta_Solana2022a_6c_nfs_allfeat.parquet')
# df = pd.read_parquet('/home/mpaul/projects/enta_dl/enta_workspace/src/enta_ml/data/test_df_mobile_tc_vs_solana_twc_nfs_allSrc_40apps_6.parquet')
# df = pd.read_csv('/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/2023c_Mobile_LTE_nfs_extract.csv')
df = pd.read_csv('/home/mpaul/projects/mpaul/mai/data/jan09/dalhousie_data_nfs_extract_jan09.csv')


for ind in df.index:
    value1 = df['splt_ps'][ind]
    print(type(value1))
    print(value1)
    value1 = value1.strip('[')
    value1= value1.strip(']')
    value1 = value1.split(',')
    # print(type(value1[0]))
    value1 = list(map(lambda x: x.replace('-1', '0'), value1))
    # print(value1)
    res = [eval(i) for i in value1]
    # print(type(res[0]))
    df['splt_ps'][ind] = res
    # df['splt_ps'] = df['splt_ps'].astype(int)
    value1 = df['splt_ps'][ind]
    print(type(value1[0]))

    value1 = df['splt_direction'][ind]
    value1 = value1.strip('[')
    value1 = value1.strip(']')
    value1 = value1.split(',')
    value1 = list(map(lambda x: x.replace('-1', '0'), value1))
    res = [eval(i) for i in value1]
    df['splt_direction'][ind] = res
    # value1 = df['splt_direction'][ind]

    value1 = df['splt_piat'][ind]
    value1 = value1.strip('[')
    value1 = value1.strip(']')
    value1 = value1.split(',')
    value1 = list(map(lambda x: x.replace('-1', '0'), value1))
    res = [eval(i) for i in value1]
    df['splt_piat'][ind] = res
    # value1 = df['splt_piat'][ind]


# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# train_df.to_parquet('/home/mpaul/projects/enta_dl/enta_workspace/enta_data/train_test_data/final_processed_data/train_v2.parquet')
# test_df.to_parquet('/home/mpaul/projects/enta_dl/enta_workspace/enta_data/train_test_data/final_processed_data/test_v2.parquet')

# df.to_csv('/home/mpaul/projects/enta_dl/enta_workspace/enta_data/train_test_data/dalhousie_nims_all_dataset_v3.csv')

# df.to_csv('/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/2023c_Mobile_LTE_nfs_extract_v3.csv')
df.to_csv('/home/mpaul/projects/mpaul/mai/data/jan09/dalhousie_data_nfs_extract_jan09_v2.csv')
 