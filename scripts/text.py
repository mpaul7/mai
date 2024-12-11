import pandas as pd
import numpy as np
import glob
import os

from sklearn.model_selection import train_test_split


col = ['sip', 'sport', 'dip', 'dport', 'proto', 'first_timestamp', 'total_time', 'sni', 'pkt_fwd_count', 'pl_fwd_count',
      'last_timestamp_fwd', 'pl_len_fwd_mean', 'pl_len_fwd_stdev', 
       'pl_len_fwd_total', 'pl_len_fwd_min', 'pl_len_fwd_max',
       'pkt_len_fwd_mean', 'pkt_len_fwd_stdev', 'pkt_len_fwd_total',
       'pkt_len_fwd_min', 'pkt_len_fwd_max', 'iat_fwd_mean', 'iat_fwd_stdev',
       'iat_fwd_total', 'iat_fwd_min', 'iat_fwd_max', 'pkt_bwd_count',
       'pl_bwd_count', 'last_timestamp_bwd', 'pl_len_bwd_mean',
       'pl_len_bwd_stdev', 'pl_len_bwd_total', 'pl_len_bwd_min',
       'pl_len_bwd_max', 'pkt_len_bwd_mean', 'pkt_len_bwd_stdev',
       'pkt_len_bwd_total', 'pkt_len_bwd_min', 'pkt_len_bwd_max',
       'iat_bwd_mean', 'iat_bwd_stdev', 'iat_bwd_total', 'iat_bwd_min',
       'iat_bwd_max', 'dd', 'dn', 'dns', 'ds', 'application', 'app',
       'traffic_type']

# all_files = glob.glob(f'/home/mpaul/projects/mpaul/mai/data/dns/twc_extracted_data/*.csv', recursive=True)

def bucketize_data(bucket_size=30, df=None):
    df = df[(df['sport'] == 53) | (df['dport'] == 53)]
    print(df.shape)
    # Step 1: Convert `first_timestamp` to a datetime format if it's not already
    df['first_timestamp'] = pd.to_datetime(df['first_timestamp'])
    df = df.sort_values(by='first_timestamp', ascending=True)
    df['time_difference'] = df['first_timestamp'] - df['first_timestamp'].iloc[0]
    df['time_difference_seconds'] = df['time_difference'].dt.total_seconds()

    # bucket_size = 30
    max_time = df['time_difference_seconds'].max()
    bins = list(range(0, int(max_time) + bucket_size, bucket_size))

    # Use pd.cut to assign each row to a bucket based on the time difference in seconds
    df['bucket'] = pd.cut(
        df['time_difference_seconds'], 
        bins=bins, 
        labels=range(len(bins)-1),
        right=True
    )
    output_col = ['sip', 'sport', 'dip', 'dport', 'proto', 'bucket', 'first_timestamp', 'time_difference_seconds','total_time', 'sni', 
                'pkt_fwd_count', 'pl_fwd_count','last_timestamp_fwd',  
                'filename'
                ]
    
    df.to_csv('/home/mpaul/projects/mpaul/mai/data/analysis/2023a_Wireline_Ethernet.csv', columns=output_col)
    # df.to_csv('/home/mpaul/projects/mpaul/mai/data/analysis/videoStream_youtube_00000000_Test2023a_Wireline_Ethernet_1.csv', columns=output_col)
    # # Step 3: Calculate aggregate features within each bucket
    # # Define aggregations for the required features
    aggregated_df = df.groupby(['bucket']).agg(
        Average_fw_packet_size=('pkt_len_fwd_mean', 'mean'),
        Average_bw_packet_size=('pkt_len_bwd_mean', 'mean'),
        Average_fw_total_pl_bytes=('pl_len_fwd_total', 'mean'),
        Average_bw_total_pl_bytes=('pl_len_bwd_total', 'mean'),
        fw_flow_count=('pkt_fwd_count', 'sum'),
        bw_flow_count=('pkt_bwd_count', 'sum')
    ).reset_index()

    # # Step 4: Compute the ratio of total backward flows to forward flows
    # # Note: We use np.where to handle division by zero gracefully
    aggregated_df['Ratio_of_total_bw_flows_and_fw_flows'] = np.where(
        aggregated_df['fw_flow_count'] != 0,
        aggregated_df['bw_flow_count'] / aggregated_df['fw_flow_count'],
        np.nan  # Assign NaN where fw_flow_count is zero
    )

    # # Optional: Drop intermediate columns used for calculating the ratio
    aggregated_df = aggregated_df.drop(columns=['fw_flow_count', 'bw_flow_count'])

    # # Display the resulting aggregated DataFrame
    # print(aggregated_df)

    aggregated_df = aggregated_df.dropna()
    aggregated_df.to_csv('/home/mpaul/projects/mpaul/mai/data/analysis/aggregated_2023a_Wireline_Ethernet.csv')
    return aggregated_df

if __name__ == '__main__':
    
    
    dns = [
        # '/home/mpaul/projects/mpaul/mai/data/dns/twc_extracted_data/2023a_Wireline_Ethernet.csv',
                # '/home/mpaul/projects/mpaul/mai/data/dns/twc_extracted_data/2024a_Wireline_Ethernet.csv',
                # '/home/mpaul/projects/mpaul/mai/data/dns/twc_extracted_data/Test2023a_Wireline_Ethernet.csv',
                # '/home/mpaul/projects/mpaul/mai/data/dns/twc_extracted_data/Test2024a_Wireline_Ethernet.csv',
                # '/home/mpaul/projects/mpaul/mai/data/dns/twc_output/videoStream_amazonPrimeVideo_20231212_Solana2023a_1min_0.csv',
                # '/home/mpaul/projects/mpaul/mai/data/videoStream_youtube_00000000_Test2023a_Wireline_Ethernet_1_20241113150342.csv',
                '/home/mpaul/projects/mpaul/mai/data/videoStream_netflix_00000000_Test2023a_Wireline_Ethernet_1_20241113150742.csv'
                ]
    dns_attack = ['/home/mpaul/projects/mpaul/mai/data/dns/twc_extracted_data/dns_attack_amplification.csv',
                '/home/mpaul/projects/mpaul/mai/data/dns/twc_extracted_data/dns_attack_nxdomain.csv'
                        ]
    bucket_size = 30
    merged_df = pd.concat([pd.read_csv(file) for file in dns], ignore_index=True)
    label = 'dns'
    
    # for file in all_files:
    
    # head, tail = os.path.split(file)
    bucketize_df = bucketize_data(bucket_size=bucket_size, df=merged_df)
    bucketize_df['label'] = label
    # output = os.path.join('/home/mpaul/projects/mpaul/mai/data/dns/bucketized_data', f'{label}_{bucket_size}s_data.csv')  
    # bucketize_df.to_csv(output, index=False) 
    
    # train_df, test_df = train_test_split(bucketize_df, test_size=0.3, random_state=42)
    # train_df.to_csv(os.path.join('/home/mpaul/projects/mpaul/mai/data/dns/bucketized_data', f'{label}_{bucket_size}s_train_data.csv'), index=False)
    # test_df.to_csv(os.path.join('/home/mpaul/projects/mpaul/mai/data/dns/bucketized_data', f'{label}_{bucket_size}s_test_data.csv'), index=False)   
