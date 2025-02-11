import pandas as pd 
import ast
import numpy as np
import os

def process_sequence_feature(x, replace_value=-1, new_value=0):
    """
    Process sequence features by converting string to list and replacing values
    Args:
        x: Input string or list
        replace_value: Value to be replaced
        new_value: New value to replace with
    """
    try:
        if isinstance(x, str):
            values = ast.literal_eval(x)
        else:
            values = x
        return [new_value if i == replace_value else float(i) for i in values]
    except (ValueError, SyntaxError) as e:
        print(f"Error processing value {x}: {str(e)}")
        return []

def process_dataframe(df):
    """Process all sequence features in the dataframe"""
    sequence_features = ['splt_direction', 'splt_piat', 'splt_ps']
    
    for feature in sequence_features:
        if feature in df.columns:
            print(f"Processing {feature}...")
            df[f'{feature}_original'] = df[feature].copy()  # Keep original data
            df[feature] = df[feature].apply(process_sequence_feature)
    
    return df

def save_dataframe(df, output_path, name):
    """Save dataframe in both CSV and Parquet formats"""
    base_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    
    # Create output paths
    csv_path = os.path.join(base_dir, f"{base_name}_{name}_processed.csv")
    parquet_path = os.path.join(base_dir, f"{base_name}_{name}_processed.parquet")
    
    try:
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        print(f"Saved processed files:\n- {csv_path}\n- {parquet_path}")
    except Exception as e:
        print(f"Error saving files: {str(e)}")

def main():
    # File paths
    file_paths = {
        'mobile': '/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/2023c_Mobile_LTE_nfs_extract.csv',
        'youtube': '/home/mpaul/projects/mpaul/mai/data/solana_2023c_data_nfs_extract/pcaps_nfs_extract_youtube.csv'
    }
    
    # Process each dataset
    processed_dfs = {}
    for name, path in file_paths.items():
        try:
            print(f"\nProcessing {name} dataset...")
            df = pd.read_csv(path)
            processed_dfs[name] = process_dataframe(df)
            
            # Print sample verification
            print(f"\nVerification for {name} dataset:")
            print(f"Shape: {processed_dfs[name].shape}")
            for feature in ['splt_direction', 'splt_piat', 'splt_ps']:
                if feature in processed_dfs[name].columns:
                    sample = processed_dfs[name][feature].iloc[0]
                    print(f"{feature} sample: {sample[:5]}...")  # Show first 5 elements
            
        except Exception as e:
            print(f"Error processing {name} dataset: {str(e)}")
    
    # Concatenate all processed dataframes
    try:
        print("\nConcatenating datasets...")
        combined_df = pd.concat(processed_dfs.values(), ignore_index=True)
        print(f"Combined shape: {combined_df.shape}")
        
        # Save combined dataset
        output_path = file_paths['mobile']  # Using mobile path as base
        save_dataframe(combined_df, output_path, 'combined')
        
        # Print final verification
        print("\nFinal combined dataset info:")
        print(combined_df.info())
        
    except Exception as e:
        print(f"Error during concatenation or saving: {str(e)}")

if __name__ == "__main__":
    main()

