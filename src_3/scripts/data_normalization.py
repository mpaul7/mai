import pandas as pd
import numpy as np

def normalize_sequence_features(df, features):
    """
    Normalize sequence features in a DataFrame by dividing by the maximum absolute value.
    
    Args:
        df: pandas DataFrame containing the sequence features
        features: list of feature names to normalize
    
    Returns:
        DataFrame with normalized sequence features
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_normalized = df.copy()
    
    for feature in features:
        # Get all sequences for this feature
        sequences = df[feature].values
        print(sequences, 88888)
        for seq in sequences:
            print(seq)
        # Flatten all sequences into a single arrayq
        all_values = np.concatenate([seq for seq in sequences])
        # print(all_values, 99999)
        
        # Find the maximum absolute value
        max_abs_val = np.max(np.abs(all_values))
        
        # Normalize each sequence by dividing by max_abs_val
        if max_abs_val != 0:  # Avoid division by zero
            normalized_sequences = [seq / max_abs_val for seq in sequences]
        else:
            normalized_sequences = sequences
            
        # Store normalized sequences back in the DataFrame
        df_normalized[feature] = normalized_sequences
    
    return df_normalized

    
    
    
if __name__ == "__main__":
    # df  = pd.read_parquet("/home/mpaul/projects/mpaul/mai/data/train_dalhousie_nims_7app_nfs_stat_features_v2.parquet")
    df = pd.read_csv("/home/mpaul/projects/mpaul/mai/data/test_solana_7apps_nfs_v4.csv")
    print(df.columns)
    # features = ["splt_ps", "splt_piat", "splt_direction"]
    # Example usage:
    features = [
        # "splt_ps", "splt_piat", 
        "stat_features"
                ]
    df_normalized = normalize_sequence_features(df, features)
    
    # Normalize statistical features
    stat_features = [
    'bidirectional_duration_ms', 'bidirectional_packets',
    'bidirectional_bytes', 'src2dst_duration_ms', 'src2dst_packets',
    # 'src2dst_bytes', 'dst2src_duration_ms', 'dst2src_packets',
    # 'dst2src_bytes', 'bidirectional_min_ps', 'bidirectional_mean_ps',
    # 'bidirectional_stddev_ps', 'bidirectional_max_ps', 'src2dst_min_ps',
    # 'src2dst_mean_ps', 'src2dst_stddev_ps', 'src2dst_max_ps',
    # 'dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_stddev_ps',
    # 'dst2src_max_ps', 'bidirectional_min_piat_ms',
    # 'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms',
    # 'bidirectional_max_piat_ms', 'src2dst_min_piat_ms',
    # 'src2dst_mean_piat_ms', 'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms',
    # 'dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms',
    # 'dst2src_max_piat_ms', 'bidirectional_syn_packets',
    # 'bidirectional_cwr_packets', 'bidirectional_ece_packets',
    # 'bidirectional_urg_packets', 'bidirectional_ack_packets',
    # 'bidirectional_psh_packets', 'bidirectional_rst_packets',
    # 'bidirectional_fin_packets', 'src2dst_syn_packets',
    # 'src2dst_cwr_packets', 'src2dst_ece_packets', 'src2dst_urg_packets',
    # 'src2dst_ack_packets', 'src2dst_psh_packets', 'src2dst_rst_packets',
    # 'src2dst_fin_packets', 'dst2src_syn_packets', 'dst2src_cwr_packets',
    # 'dst2src_ece_packets', 'dst2src_urg_packets', 'dst2src_ack_packets',
    # 'dst2src_psh_packets', 'dst2src_rst_packets', 'dst2src_fin_packets'
]

    # Create copy of DataFrame for statistical features normalization
    df_stat_normalized = df.copy()

    for feature in stat_features:
        # Get values for this statistical feature
        values = df[feature].values
        
        # Find max absolute value for normalization
        max_abs_val = np.max(np.abs(values))
        
        # Normalize by dividing by max_abs_val
        if max_abs_val != 0:  # Avoid division by zero
            df_stat_normalized[feature] = values / max_abs_val
        else:
            df_stat_normalized[feature] = values
            
    # Merge the normalized sequence features with normalized statistical features
    # Add network metadata columns from original DataFrame
    metadata_cols = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol']
    df_normalized = pd.concat([df[metadata_cols], 
                            df_normalized[features], 
                            df['splt_direction'],
                            df_stat_normalized[stat_features], 
                            df['label']], 
                            axis=1)
    # df_normalized = pd.concat([df_normalized[features], df_stat_normalized[stat_features], df_normalized['label'], 
    #                            df_normalized['splt_direction']], axis=1)

    # Check the results
    for feature in features:
        original_seq = df[feature].iloc[0]
        normalized_seq = df_normalized[feature].iloc[0]
        
        print(f"\n{feature}:")
        print(f"Original first few values: {original_seq[:5]}")
        print(f"Normalized first few values: {normalized_seq[:5]}")
        print(f"Min value: {min(normalized_seq)}")
        print(f"Max value: {max(normalized_seq)}")
        
        print(df_normalized.columns)
        # df_normalized.to_parquet("/home/mpaul/projects/mpaul/mai/data/train_dalhousie_nims_7app_nfs_stat_features_v2_normalized.parquet")
        # df_normalized.head(200).to_csv("/home/mpaul/projects/mpaul/mai/data/train_dalhousie_nims_7app_nfs_stat_features_v2_normalized.csv")
        df_normalized.to_csv("/home/mpaul/projects/mpaul/mai/data/test_solana_7apps_nfs_v4_normalized.csv")

        # print(df_normalized.head(10))
        
        
        