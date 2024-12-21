import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model_validator import ModelValidator

# Example test model (users will replace this with their own model)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 8, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 8, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.MaxPool2d(2,2),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(2,2),
                nn.Conv2d(16, 20, 3),
                nn.ReLU(),
                nn.Conv2d(20, 10, 3),
                nn.AvgPool2d(3, 2),
                )

    def forward(self, x):
        x = self.conv(x)
        
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x

def test_model_validation():
    model = Net()
    validator = ModelValidator(model)
    
    assert validator.count_parameters(), "Parameter count check failed"
    assert validator.check_batch_norm(), "Batch normalization check failed"
    assert validator.check_gap(), "GAP check failed"
    assert validator.run_all_checks(), "Overall validation failed" 