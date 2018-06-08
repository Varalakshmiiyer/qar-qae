import pandas as pd
from utils import read_json
import h5py
import numpy as np

def save_data(data, output_file):
	# print 'Data items: [%d]' % (len(data))
	df = pd.DataFrame(data)
	df.to_csv(output_file)
	return df
