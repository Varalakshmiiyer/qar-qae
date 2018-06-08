import pandas as pd

base_dir = '/sb-personal/cvqa/results/c2vqa-verbs-results-final/'
stats_file = base_dir + 'stats.csv'
# stats_file = '/sb-personal/cvqa/results/c2vqa-verbs-results/stats.csv'
stats_df = pd.read_csv(stats_file)

print(stats_df.groupby(['model']).mean())

# df = pd.read_csv(base_dir + 'qcatt-0-test_results.csv')
# filtered_df = df[df['y_true'] != df['y_predict']]
# # filtered_df = filtered_df[filtered_df['y_true'] != "RELEVANT"]
# filtered_df = filtered_df[filtered_df['y_predict'] != "RELEVANT"]
# print filtered_df[['y_true','y_predict','caption','question']].sample(40)
