import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import re

# Reference Colab Link https://colab.research.google.com/drive/1AU1q0CsYv0SUjHkzDRaJISpxT2zlnPkv#scrollTo=jDzxdjnn16ue

def load_data(FILE_PATH):
    df = pd.read_csv(FILE_PATH)
    cols_to_drop = [
    'Unnamed: 0',
    'user',
    'collection_logical_name',
    'start_after_collection_ids',
    'collection_id',
    'machine_id',
    'alloc_collection_id',
    'collection_name',
    'constraint'
    ]
    df.drop(cols_to_drop, axis=1, inplace=True)
    df = df[df['time'] >= 600 * 10**6].copy()

    # Add 'offset' column
    df['offset'] = 600 * 10**6

    # Calculate corrected time and convert to seconds
    df['timeCorr'] = (df['time'] - df['offset']) / 10**6

    # Calculate hours
    df['hours'] = df['timeCorr'] / 3600

    # Cast hours to integers
    df['hours'] = df['hours'].astype(int)

    # Calculate hour bins
    df['hourBin'] = df['hours'] % 24

    # Calculate time difference in hours and drop original time columns
    df['time_diff_hrs'] = ((df['end_time'] - df['start_time']) / (10**6 * 3600))
    df = df.drop(columns=['start_time', 'end_time'])
    # Convert 'timeCorr' to datetime format
    df['timeCorr'] = pd.to_datetime(df['timeCorr'])

    # Set 'timeCorr' as the index
    df.set_index('timeCorr', inplace=True)
    return df

def handle_missing_values(df):
    df = df.fillna(0)
    df['vertical_scaling'] = df['vertical_scaling'].astype(int)
    df['scheduler'] = df['scheduler'].astype(int)
    return df

def normalize_df(df):
    # Calculate maximum and minimum values for normalization
    time_diff_max = df['time_diff_hrs'].max()
    time_diff_min = df['time_diff_hrs'].min()
    assigned_max = df['assigned_memory'].max()
    assigned_min = df['assigned_memory'].min()
    priority_max = df['priority'].max()
    priority_min = df['priority'].min()

    # Normalize selected columns
    df['priority'] = (df['priority'] - priority_min) / priority_max
    df['assigned_memory'] = (df['assigned_memory'] - assigned_min) / assigned_max
    df['time_diff_hrs'] = (df['time_diff_hrs'] - time_diff_min) / time_diff_max
    return df

def preprocessing_pipeline(df):
    numeric_features = ['time_diff_hrs', 'assigned_memory', 'priority']
    numerical_cols = [x for x in list(df.columns) if str(df[x].dtype)!="object"]
    categorical_cols = [x for x in df.columns if x not in numerical_cols]
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())])

    categorical_features = categorical_cols
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
    
    # Convert categorical labels to numerical indices using LabelEncoder
    label_encoder = LabelEncoder()
    df['eventIndex'] = label_encoder.fit_transform(df['event'])
    return df

def split_cpu_memory_values(df):
    def split_cell(df):
        number = re.compile(r"\s*?(\d+(\.\d*)?|\.\d+)\s*")
        if type(df) !=float:
            df = df.split(":")
            cpu = df[1]
            cpu = float(cpu.split(",")[0])
            memory = df[2].split("}")
            if number.match(memory[0]):
                memory = float(memory[0])
            # print("***", memory)
            else:
                memory = np.nan
        else:
            cpu, memory = np.nan, np.nan

        return [cpu, memory]
    
    cpu_memory_cols = ['resource_request',
    'average_usage',
    'maximum_usage',
    'random_sample_usage']

    for i in cpu_memory_cols:
        df[i+"_cpu"] = df[i].apply(lambda x: split_cell(x)[0])
        df[i+"_memory"] = df[i].apply(lambda x: split_cell(x)[1])
        df.drop(i, inplace=True, axis=1)

    return df

def handle_distributions(df):
    df=df.fillna(0)
    # Convert the string arrays to actual arrays
    df['cpu_usage_distribution'] = df['cpu_usage_distribution'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
    df['tail_cpu_usage_distribution'] = df['tail_cpu_usage_distribution'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

    # Calculate the summary statistics
    df['cpu_mean'] = df['cpu_usage_distribution'].apply(lambda x: np.mean(x))
    df['cpu_std_dev'] = df['cpu_usage_distribution'].apply(lambda x: np.std(x))
    df['cpu_median'] = df['cpu_usage_distribution'].apply(lambda x: np.median(x))

    df['tail_cpu_mean'] = df['tail_cpu_usage_distribution'].apply(lambda x: np.mean(x))
    df['tail_cpu_std_dev'] = df['tail_cpu_usage_distribution'].apply(lambda x: np.std(x))
    df['tail_cpu_median'] = df['tail_cpu_usage_distribution'].apply(lambda x: np.median(x))

    # Drop the original columns
    columns_to_drop = ['cpu_usage_distribution', 'tail_cpu_usage_distribution']
    df = df.drop(columns=columns_to_drop)
    df=df.drop('random_sample_usage_memory',axis=1)
    return df

def preprocessing(FILE_PATH):
    df = load_data(FILE_PATH)
    cleaned_df = handle_missing_values(df)
    normalized_df = normalize_df(cleaned_df)
    preprocessed_df = preprocessing_pipeline(normalized_df)
    split_df = split_cpu_memory_values(preprocessed_df)
    final_df = handle_distributions(split_df)
    return final_df
    # return preprocessed_df
