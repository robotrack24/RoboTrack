# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
CoTracker training infrastructure using pure torch.distributed.
Replaces PyTorch Lightning Lite with explicit DDP control.
"""
