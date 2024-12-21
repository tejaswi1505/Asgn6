import pytest
import torch
import torch.nn as nn
from model_validator import ModelValidator

# Example test model (users will replace this with their own model)
class SampleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def test_model_validation():
    model = SampleNN()
    validator = ModelValidator(model)
    
    assert validator.count_parameters(), "Parameter count check failed"
    assert validator.check_batch_norm(), "Batch normalization check failed"
    assert validator.check_dropout(), "Dropout check failed"
    assert validator.check_final_layer(), "Final layer check failed"
    assert validator.run_all_checks(), "Overall validation failed" 