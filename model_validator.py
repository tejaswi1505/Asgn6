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
        """Check if model has parameter count less than 20000"""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.validation_results['parameter_count'] = total_params
        is_valid =  total_params < 20000
        logger.info(f"Total parameters: {total_params}")
        return is_valid

    def check_batch_norm(self) -> bool:
        """Verify if model uses batch normalization"""
        has_batch_norm = any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                           for module in self.model.modules())
        self.validation_results['has_batch_norm'] = has_batch_norm
        logger.info(f"Has batch normalization: {has_batch_norm}")
        return has_batch_norm

    def check_gap(self) -> bool:
        """Verify if model uses dropout"""
        has_gap = any(isinstance(module, nn.AvgPool2d) 
                        for module in self.model.modules())
        self.validation_results['has_gap'] = has_gap
        logger.info(f"Has GAP: {has_gap}")
        return has_gap

    def run_all_checks(self) -> bool:
        """Run all validation checks"""
        checks = [
            self.count_parameters(),
            self.check_batch_norm(),
            self.check_gap()
        ]
        return all(checks)