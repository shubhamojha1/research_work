import numpy as np
import pandas as pd
import math
import csv
from utils.preprocessing import preprocessing
from sklearn.preprocessing import MinMaxScaler
import torch
from quaesita.TimeSeriesDataset import timeseriesDatasetCreateBatch 

# # code to be put into main.py


FILE_PATH = './data/borg_traces_data.csv'
PREPROCESSED_FILE_PATH = './data/borg_traces_data_preprocessed_10.csv'

# # ########## RUN ONCE ONLY ##########
# # df = preprocessing(FILE_PATH)
# # df.to_csv(PREPROCESSED_FILE_PATH)
# # ########## RUN ONCE ONLY ##########

# # original paper takes job arrival rates (JARs)

# # with open(PREPROCESSED_FILE_PATH, 'r') as csvfile:
# #     reader = csv.DictReader(csvfile)
# #     # print(reader)
# #     count = 0
# #     testNo = 0
# #     dataNpArray = [] #np.empty([0, 2], int)
# #     scaleUp = 1
# #     for row in reader:
# #         # print(row)
# #         # jobCount = float(row['JobCount'])//float(scaleUp)
# #         # scheduling_class = row['scheduling_class']
# #         timeCorr = row['timeCorr']
# #         instance_events_type = row['instance_events_type']
# #         count = int(count) + 1
# #         testNo = int(testNo) + 1
# #         # dataNpArray = np.append(dataNpArray, [[count, jobCount]], axis=0)
# #         dataNpArray.extend([[count, timeCorr, instance_events_type]])
# #         # dataNpArray = np.append(dataNpArray, [[count, timeCorr, instance_events_type]], axis=0)
# #         break

# # index = [str(i) for i in range(1, len(dataNpArray) + 1)]
# # data_df = pd.DataFrame(dataNpArray, index=index, columns=['timePeriod', 'timeCorr', 'instance_events_type'])
# # print("********")
# # print(data_df)


data_df=pd.read_csv(PREPROCESSED_FILE_PATH)
data_df.reset_index(drop=True, inplace=True)
data_df.drop(['timeCorr', 'event'], inplace=True, axis=1)
#-----> need to handle 'event' column later

# job_arrival_count = data_df[['jobCount']].values.astype('float32')
job_arrival_count = data_df.values.astype('float32')
#-----> need to remove 'y' values from data_df


# # # dataset_name = in_workload_key


# # min-max scaling 
scaler = MinMaxScaler(feature_range=(-1,1))

# # create test/validation/test 60-15-15 split
split_size = 20 # to create 80-20 split for the training/testing dataset
test_data_size = int(math.ceil(len(job_arrival_count) * split_size / 100))

_train_data1 = job_arrival_count[:-test_data_size]
_test_data = job_arrival_count[-test_data_size:]

cv_data_size = int(math.ceil(len(_train_data1) * 0.25))
_train_data = _train_data1[:-cv_data_size]
_cross_val_data = _train_data1[-cv_data_size:]

test_len = len(_test_data)
print(len(_train_data), len(_cross_val_data), len(_test_data))

_train_data = scaler.fit_transform(_train_data.reshape(-1,1))
_test_data = scaler.fit_transform(_test_data.reshape(-1,1))
_cross_val_data = scaler.fit_transform(_cross_val_data.reshape(-1,1))

# convert data to 1D tensor
_train_data = torch.FloatTensor(_train_data)
_cross_val_data = torch.FloatTensor(_cross_val_data)
_test_data = torch.FloatTensor(_test_data)
    
lookback_set = np.round(np.arange(1,9,1) * 0.1 * test_len)         
# print(lookback_set)
# experiment_params = [[WINDOW_SIZE, BATCH_SIZE]]
# print(_train_data)
_window_size=1
_batch_size=1024

seq2seq_dataset_params = {
        'window_size' : int(_window_size),   # input sequence length
        'target_stride' : 1,   # predict number of target steps
        'batch_size' : int(_batch_size),
        'flag'  :   False
    }
train_datasetA = timeseriesDatasetCreateBatch(_train_data, **seq2seq_dataset_params)  

for x, y, tgt_msk in train_datasetA:
    # print(x, y, tgt_msk)
    print("****"*10)
    print(x)

# # results = np.empty([0,23],str)

# # for _window_size, _batch_size in experiment_params:
# #     results = np.append(results, _results, axis = 0)
# #     _results = call_main(_window_size, _batch_size, _train_data, _cross_val_data, _test_data, GPU)#sys.argv[4])            
# #     # <----- call to the main func here

# # index = [str(i) for i in range(1, len(results) + 1)]
# # data_df = pd.DataFrame(results, index=index, columns=['Dataset','Epoch','learning_rate','Cost function','Window Size','Batch Size', 'Dropout',
# #                                                                   'd_model','nhead','Train MAPE','Train RMSE','Train MASE','CV MAPE','CV RMSE','CV MASE','Test MAPE','Test RMSE', 'Test MASE', 'GPU Name',
# #                                                                   'training-time','train-inference-time','cv-inference-time','test-inference-time'
# #                                                                   ])
# # curr_file = pd.read_csv(EXP_FOLDER_PATH + str(DATASET_NAME)+".csv", usecols=['Dataset','Epoch','learning_rate','Cost function','Window Size','Batch Size', 'Dropout',
# #                                                                   'd_model','nhead','Train MAPE','Train RMSE','Train MASE','CV MAPE','CV RMSE','CV MASE','Test MAPE','Test RMSE', 'Test MASE', 'GPU Name',
# #                                                                   'training-time','train-inference-time','cv-inference-time','test-inference-time'
# #                                                                   ])

# # data_df = data_df.append(curr_file)
# # data_df.to_csv(EXP_FOLDER_PATH + str(DATASET_NAME)+".csv")

# a = ['Dataset','Epoch','learning_rate','Cost function','Window Size','Batch Size', 'Dropout',
#                                     'd_model','nhead','Train MAPE','Train RMSE','Train MASE','CV MAPE','CV RMSE','CV MASE','Test MAPE','Test RMSE', 'Test MASE', 'GPU Name',
#                                                                   'training-time','train-inference-time','cv-inference-time','test-inference-time'
#                                                                   ]
# b = ['Dataset','Epoch','learning_rate','Cost function','Window Size','Batch Size', 'Dropout',
#                                                                   'd_model','nhead','Train MAPE','Train RMSE','Train MASE','CV MAPE','CV RMSE','CV MASE','Test MAPE','Test RMSE', 'Test MASE', 'GPU Name',
#                                                                   'training-time','train-inference-time','cv-inference-time','test-inference-time'
#                                                                   ]
# print(a==b)
