#!/usr/bin/env python

import wget
import os
url_root = 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosei/seq_length_20/'
datas = ['audio_test.h5', 'audio_train.h5', 'audio_valid.h5']
# labels = ['mosi2uni_Test_labels_5class.csv', 'mosi2uni_Train_labels_5class.csv']
elabels = ['ey_test.h5', 'ey_train.h5', 'ey_valid.h5']
labels = ['y_test.h5', 'y_train.h5', 'y_valid.h5']

os.makedirs('./MOSEI/', exist_ok=True)

for file_name in datas + labels:
  if os.path.exists('./MOSEI/' + file_name):
    continue
  url = url_root + 'data/' + file_name
  out_path = './MOSEI/'+file_name
  wget.download(url, out=out_path)
print(os.listdir('./MOSEI/'))

