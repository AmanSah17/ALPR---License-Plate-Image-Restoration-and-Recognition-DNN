import os
import glob
import torch
from pathlib import Path
from omegaconf import OmegaConf
from models.model_registry import ModelRegistry, initialize_registry
from engine.composite_lightning import CompositeRestorationLightningModule

def export_all():
    out_dir = Path("outputs/visualizations/onnx")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    initialize_registry()
    
    # Map model names to their best checkpoint paths
    best_checkpoints = {}
    exp_dir = Path("outputs/experiments")
    
    print("Scanning for trained models...")
    for model_name in ModelRegistry.list_models().keys():
        # Find all experiment folders for this model
        # We skip 'hybrid_attention' as per user request (currently training)
        if "hybrid_attention" == model_name:
            continue
            
        folders = list(exp_dir.glob(f"*_train_{model_name}"))
        if not folders:
            continue
            
        # Get the latest folder
        latest_folder = sorted(folders)[-1]
        
        # Get the best checkpoint
        ckpt_files = list(latest_folder.glob("checkpoints/best-epoch=*.ckpt"))
        if ckpt_files:
            # Sort by val_psnr inside the filename or just take the last modified
            best_ckpt = max(ckpt_files, key=lambda x: os.path.getmtime(x))
            best_checkpoints[model_name] = best_ckpt
            print(f"Found {model_name}: {best_ckpt.name}")

    if not best_checkpoints:
        print("No trained models found with checkpoints.")
        return

    for model_name, ckpt_path in best_checkpoints.items():
        print(f"\nExporting {model_name} to ONNX...")
        try:
            # Build base model
            model = ModelRegistry.build(model_name)
            
            # Load weights
            cfg = OmegaConf.create({
                "losses": {"l1_weight": 1.0, "lpips_weight": 0.0},
                "dataset": {"input_mode": "center_upscale", "augmentation": {"enabled": False}},
                "model": {"name": model_name}
            })
            
            lightning_module = CompositeRestorationLightningModule.load_from_checkpoint(
                str(ckpt_path), 
                model=model,
                cfg=cfg,
                strict=False
            )
            model = lightning_module.model.cuda()
            model.eval()
            
            channels = 15 if "spatiotemporal" in model_name else 3
            dummy_input = torch.randn(1, channels, 64, 256).cuda()
            
            out_file = out_dir / f"{model_name}.onnx"
            torch.onnx.export(
                model,
                dummy_input,
                str(out_file),
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['input_image'],
                output_names=['restored_image'],
                dynamic_axes={
                    'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'restored_image': {0: 'batch_size', 2: 'height', 3: 'width'}
                }
            )
            print(f"✓ Saved to {out_file}")
            
        except Exception as e:
            print(f"Failed to export {model_name}: {e}")

if __name__ == "__main__":
    export_all()
