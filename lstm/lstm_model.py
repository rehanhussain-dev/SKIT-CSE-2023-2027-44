import torch
import torch.nn as nn


class YieldLSTM(nn.Module):
    """
    LSTM branch for crop-growth trajectory modelling.

    Input shape:
        (batch_size, sequence_length, input_size)

    Output:
        One prediction for each sequence.
    """

    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: (batch, sequence, features)
        """

        lstm_output, _ = self.lstm(x)

        # Use the final timestep representation
        final_output = lstm_output[:, -1, :]

        prediction = self.fc(final_output)

        return prediction