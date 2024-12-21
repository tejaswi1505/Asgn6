import torch
import torch.nn as nn
from typing import Type
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self, model: Type[nn.Module]):
        self.model = model
        self.validation_results = {}

    def count_parameters(self) -> bool:
        """Check if model has reasonable parameter count (between 10k and 100M)"""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.validation_results['parameter_count'] = total_params
        is_valid = 10000 <= total_params <= 100_000_000
        logger.info(f"Total parameters: {total_params}")
        return is_valid

    def check_batch_norm(self) -> bool:
        """Verify if model uses batch normalization"""
        has_batch_norm = any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                           for module in self.model.modules())
        self.validation_results['has_batch_norm'] = has_batch_norm
        logger.info(f"Has batch normalization: {has_batch_norm}")
        return has_batch_norm

    def check_dropout(self) -> bool:
        """Verify if model uses dropout"""
        has_dropout = any(isinstance(module, nn.Dropout) 
                        for module in self.model.modules())
        self.validation_results['has_dropout'] = has_dropout
        logger.info(f"Has dropout: {has_dropout}")
        return has_dropout

    def check_final_layer(self) -> bool:
        """Check if model ends with either FC layer or Global Average Pooling"""
        modules = list(self.model.modules())
        has_valid_end = any(isinstance(modules[-1], (nn.Linear, nn.AdaptiveAvgPool2d)))
        self.validation_results['has_valid_final_layer'] = has_valid_end
        logger.info(f"Has valid final layer: {has_valid_end}")
        return has_valid_end

    def run_all_checks(self) -> bool:
        """Run all validation checks"""
        checks = [
            self.count_parameters(),
            self.check_batch_norm(),
            self.check_dropout(),
            self.check_final_layer()
        ]
        return all(checks) 