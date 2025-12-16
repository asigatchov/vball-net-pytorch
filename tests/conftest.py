import sys
import os
import torch
import pytest

# Add src to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def sample_input():
    return torch.randn(1, 15, 288, 512)