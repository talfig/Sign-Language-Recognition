# model/GestureModel.py


from CustomLSTM import *
from CustomCNNs import *


class GestureModel(nn.Module):
    def __init__(self, num_classes):
        super(GestureModel, self).__init__()
        self.cnn = CustomMobileNetV2(num_classes=num_classes)
        self.lstm = CustomLSTM(input_size=self.cnn.base_model.last_channel, hidden_size=256, num_classes=num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.size()

        # Reshape to (batch_size * seq_len, channels, height, width)
        x = x.view(batch_size * seq_len, c, h, w)

        # Pass through the CNN
        x = self.cnn(x)

        # Reshape back to (batch_size, sequence_length, features)
        x = x.view(batch_size, seq_len, -1)

        # Pass through the LSTM
        x = self.lstm(x)

        return x
