# -*- coding: utf-8 -*-
# @Time    : 2025/03/19
# @Author  : Schmetzler
# @Project : wilor_keypoints
# @FileName: __init__.py.py
import torch

class WeightsOnlyFalse():
    def __init__(self):
        self.old_kwords = torch.load.__kwdefaults__
        self.new_kwords = dict(self.old_kwords)
        self.new_kwords["weights_only"] = False

    def __enter__(self):
        torch.load.__setattr__('__kwdefaults__', self.new_kwords)

    def __exit__(self, *args, **kwargs):
        torch.load.__setattr__('__kwdefaults__', self.old_kwords)