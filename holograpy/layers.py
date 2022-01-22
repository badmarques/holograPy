import torch
import torch.nn as nn

class Classifier(nn.Module):

    def __init__(self, input_size, num_classes, dropout=0.2):
        super(Classifier, self).__init__()

        if(dropout > 0.0):
            self.layer = nn.Sequential(
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(input_size, num_classes)
            )
        else:
            self.layer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(input_size, num_classes)
            )

    def forward(self, x) -> torch.Tensor:
        return self.layer(x)
