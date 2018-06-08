import pandas as pd

stats_file = '/sb-personal/cvqa/results/argo/c2vqa-edit-verbs-results/stats.csv'
stats_df = pd.read_csv(stats_file)

print stats_df.groupby(['model']).mean()