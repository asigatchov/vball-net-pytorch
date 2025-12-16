import torch
import os
import pytest
from src.model.vballnet_v1c import VballNetV1c


def test_standard_onnx_export():
    """Test standard ONNX export functionality"""
    device = 'cpu'
    model = VballNetV1c(height=288, width=512, in_dim=15, out_dim=15).to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 15, 288, 512, device=device)
    
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            "vball_net_v1c_test.onnx",
            opset_version=18,
            input_names=["clip"],
            output_names=["heatmaps"],
            dynamic_axes={
                "clip": {0: "B"}, 
                "heatmaps": {0: "B"}
            },
            # Use traditional export instead of dynamo for GRU compatibility
            dynamo=False
        )
        print("Standard ONNX export successful!")
        
        # Check that the file was created
        assert os.path.exists("vball_net_v1c_test.onnx"), "ONNX file was not created"
        
        # Clean up the test file
        os.remove("vball_net_v1c_test.onnx")
    except Exception as e:
        pytest.fail(f"Standard ONNX export failed: {e}")


def test_stateful_onnx_export():
    """Test stateful ONNX export functionality"""
    device = 'cpu'
    model = VballNetV1c(height=288, width=512, in_dim=15, out_dim=15).to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 15, 288, 512, device=device)
    dummy_h0 = torch.randn(1, 1, 128, device=device)  # (1, B, hidden_size)
    
    try:
        torch.onnx.export(
            model,
            (dummy_input, dummy_h0),
            "vball_net_v1c_stateful_test.onnx",
            opset_version=18,
            input_names=["clip", "h0"],
            output_names=["heatmaps", "hn"],
            dynamic_axes={
                "clip": {0: "B"}, 
                "h0": {1: "B"},
                "heatmaps": {0: "B"},
                "hn": {1: "B"}
            },
            # Use traditional export instead of dynamo for GRU compatibility
            dynamo=False
        )
        print("Stateful ONNX export successful!")
        
        # Check that the file was created
        assert os.path.exists("vball_net_v1c_stateful_test.onnx"), "Stateful ONNX file was not created"
        
        # Clean up the test file
        os.remove("vball_net_v1c_stateful_test.onnx")
    except Exception as e:
        pytest.fail(f"Stateful ONNX export failed: {e}")


def test_model_path_onnx_export(tmp_path):
    """Test ONNX export when model_path is provided"""
    device = 'cpu'
    model = VballNetV1c(height=288, width=512, in_dim=15, out_dim=15).to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 15, 288, 512, device=device)
    
    # Create a temporary model path
    model_path = tmp_path / "test_model.pth"
    onnx_path = tmp_path / "test_model.onnx"
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    
    try:
        # Load the model from the path
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        
        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_path,
            opset_version=18,
            input_names=["clip"],
            output_names=["heatmaps"],
            dynamic_axes={
                "clip": {0: "B"}, 
                "heatmaps": {0: "B"}
            },
            # Use traditional export instead of dynamo for GRU compatibility
            dynamo=False
        )
        print("Model path ONNX export successful!")
        
        # Check that the ONNX file was created
        assert onnx_path.exists(), "ONNX file was not created next to model file"
    except Exception as e:
        pytest.fail(f"Model path ONNX export failed: {e}")


if __name__ == "__main__":
    test_standard_onnx_export()
    test_stateful_onnx_export()
    # test_model_path_onnx_export()  # Requires tmp_path fixture for pytest