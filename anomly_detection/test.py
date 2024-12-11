import pandas as pd

df = pd.read_csv('/home/mpaul/projects/mpaul/mai/data/dns/analysis/test_file_flat.csv')

print(df.head())

print(df.groupby('label').size())
print("\nLabel counts per bucket:")
print("=" * 40)
pd.set_option('display.max_rows', None)
bucket_label_counts = df.groupby(['bucket', 'label']).size().unstack(fill_value=0)
print(bucket_label_counts)
print("\nBuckets with attacks (dns_attack > 0):")
print("=" * 40)
attack_buckets = bucket_label_counts[bucket_label_counts['dns_attack'] > 0].index.tolist()
print(f"Bucket numbers: {attack_buckets}")

print("\nSummary statistics:")
print("=" * 40)
print("Total buckets:", len(bucket_label_counts))
print("Buckets with attacks:", len(bucket_label_counts[bucket_label_counts['dns_attack'] > 0]))
print("Buckets without attacks:", len(bucket_label_counts[bucket_label_counts['dns_attack'] == 0]))

print("\nDetailed statistics:")
print("=" * 40)
print("\nTotal flows per label:")
print(df['label'].value_counts())

