import torch
from torch import nn

class CNN(nn.Module):

    def __init__(self,
                 input_channels: int,
                 output_neurons: int,
                 state_size: int,
                 max_output_channels: int = 16,
                 drop_prob: float = 0.2):
        super().__init__()

        output_channels: int = input_channels
        min_state_size = 2 if state_size % 2 == 0 else 3

        self.conv_blocks = nn.ModuleList()

        pooled_size = state_size
        while pooled_size > min_state_size:
            input_channels = output_channels
            output_channels = min(output_channels + 1, max_output_channels)
            pooled_size //= 2

            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=2,
                              padding=1,
                              stride=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            break

        flattened_input_neurons: int = output_channels * (min_state_size ** 2)
        flattened_output_neurons: int = flattened_input_neurons * state_size

        self.linear_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= flattened_input_neurons,
                      out_features= flattened_output_neurons),
            nn.ReLU(),
            nn.Dropout(p= drop_prob),
            nn.Linear(in_features= flattened_output_neurons,
                      out_features= output_neurons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return self.linear_block(x)




