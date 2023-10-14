import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out
    

cols_to_use = ['assigned_memory', 'page_cache_memory', 
               'memory_accesses_per_instruction', 'resource_request_cpu', 
               'resource_request_memory', 'average_usage_cpu', 
               'average_usage_memory', 'maximum_usage_cpu', 
               'maximum_usage_memory', 'random_sample_usage_cpu', 
               'cpu_mean', 'cpu_std_dev', 'cpu_median', 
               'tail_cpu_mean', 'tail_cpu_std_dev', 'tail_cpu_median']


loss_fn = nn.MSELoss()
learning_rate = 0.001
num_epochs = 10
hidden_size = 50
num_layers = 2

df = pd.read_csv("../data/borg_traces_data_preprocessed.csv")

def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

for column in cols_to_use:
    model = LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    data = df[column].values
    X, y = split_sequence(data, n_steps=1)

    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.2)
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    for epoch in range(num_epochs):
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    file_name = f'{column}_losses'
    pd.DataFrame(losses, columns=['loss']).to_csv(file_name+'.csv')
    plt.plot(losses)
    plt.title(file_name)
    plt.savefig(file_name+'.png')

    
