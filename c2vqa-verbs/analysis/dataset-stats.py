import pandas as pd
import os
from utils import read_json

base_dir = '/sb-personal/cvqa/'
output_dir = os.path.join(base_dir, 'data/visual-genome/8-26-2017/generated-data/')
# questions_output_file = os.path.join(output_dir, 'specific_relevance_actions_vg_expanded_dataset-v2.csv')
questions_output_file = os.path.join(output_dir, 'editable_actions_vg_expanded_dataset-v2.csv')
caption_file = os.path.join(base_dir, 'data/cvqa/imagecaptions-vg.json')

image_captions = read_json(caption_file)

df = pd.read_csv(questions_output_file)
df = df[df['image_file'].isin(image_captions.keys())]

print 'Images: [%d]' % (len(df['image_file'].unique().tolist()))
print 'Questions: [%d]' % (len(df))
print 'Answers: [%d]' % (len(df['answer'].unique()))