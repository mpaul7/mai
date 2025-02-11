import pandas as pd 


# df = pd.read_parquet('/home/mpaul/projects/mpaul/mai/data/standard_scaler/final_train_test/test_dalhousie_nims_7app_nfs_normalized.parquet')
df = pd.read_parquet('/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/2023c_Mobile_LTE_nfs_extract_v2_normalized_v4.parquet')


for ind in df.index:
    # df['splt_piat'][ind] = df['splt_piat'][ind].tolist()
    # df['splt_ps'][ind] = df['splt_ps'][ind].tolist()
    print(type(df_dal['stat_featires'][1]))
    print(type(df['splt_piat'][ind]))
    
    # Drop the second row (index 1) from the dataframe
    df = df.drop(index=1)