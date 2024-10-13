# model/CustomLSTM.py


import torch.nn as nn


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (hn, cn) = self.lstm(x)

        # Take the output from the last time step
        x = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Fully connected layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)

        return x
