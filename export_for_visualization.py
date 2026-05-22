import torch
import argparse
import os
from models.model_registry import ModelRegistry, initialize_registry

def export_model(model_name: str, batch_size: int = 1):
    # 1. Initialize registry and build model
    initialize_registry()
    model = ModelRegistry.build(model_name)
    model.eval()
    
    # 2. Determine input shape based on model type
    # Spatiotemporal uses 15 channels (5 frames), others use 3 channels
    channels = 15 if "spatiotemporal" in model_name else 3
    input_shape = (batch_size, channels, 64, 256) # Typical plate size
    
    # 3. Create dummy data to trace the data flow
    dummy_input = torch.randn(input_shape)
    
    # 4. Export to ONNX
    output_filename = f"{model_name}_architecture.onnx"
    print(f"Tracing {model_name}...")
    
    torch.onnx.export(
        model,                      # model being run
        dummy_input,                # model input
        output_filename,            # where to save the model
        export_params=False,        # store the untrained architecture only
        opset_version=14,           # ONNX version
        do_constant_folding=True,   # optimize the graph for visualization
        input_names=['input_image'],   
        output_names=['restored_image'], 
        dynamic_axes={
            'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
            'restored_image': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"✓ Successfully exported architecture to: {os.path.abspath(output_filename)}")
    print(f"👉 Drag and drop this file into https://netron.app/ to visualize!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="spatiotemporal_hybrid_small", 
                        help="Name of the model to export (e.g., swinir_base, unet_standard)")
    args = parser.parse_args()
    
    export_model(args.model)
