import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytest

from src.model.vballnet_v1c import VballNetV1c


def test_model_compilation_and_training():
    """Test that model compiles and trains for 2 epochs on test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # For testing purposes, use CPU
    device = 'cpu'
    
    # Initialize model with 15 input frames and 15 output heatmaps
    model = VballNetV1c(height=288, width=512, in_dim=15, out_dim=15).to(device)
    
    # Create dummy dataset for testing with 15 input frames
    batch_size = 2
    num_batches = 5
    x_data = torch.randn(batch_size * num_batches, 15, 288, 512)
    y_data = torch.randn(batch_size * num_batches, 15, 288, 512)
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Simple optimizer and loss function for testing
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train for 2 epochs
    model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/2 - Loss: {epoch_loss/len(dataloader):.6f}")
    
    # Test inference/eval
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 15, 288, 512, device=device)
        test_output, _ = model(test_input)
    
    print("Eval OK. Output range:", float(test_output.min()), float(test_output.max()))
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    
    # Assertions to verify the test passed
    assert test_output.shape == (1, 15, 288, 512), f"Expected output shape (1, 15, 288, 512), got {test_output.shape}"
    assert test_output.min() >= 0.0 and test_output.max() <= 1.0, "Output values should be in range [0, 1] due to sigmoid"


if __name__ == "__main__":
    test_model_compilation_and_training()