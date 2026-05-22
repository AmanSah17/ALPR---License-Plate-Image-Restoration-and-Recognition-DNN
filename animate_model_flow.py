import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from omegaconf import OmegaConf
from models.model_registry import ModelRegistry, initialize_registry
from engine.composite_lightning import CompositeRestorationLightningModule
from datasets.rlpr_restoration_dataset import RLPRRestorationDataset
from metrics.psnr import compute_psnr
from metrics.ssim import compute_ssim

def create_animation():
    out_dir = Path("outputs/visualizations/animations")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    initialize_registry()
    best_checkpoints = {}
    exp_dir = Path("outputs/experiments")
    
    for model_name in ModelRegistry.list_models().keys():
        if "hybrid_attention" == model_name:
            continue
        folders = list(exp_dir.glob(f"*_train_{model_name}"))
        if not folders: continue
        latest_folder = sorted(folders)[-1]
        ckpt_files = list(latest_folder.glob("checkpoints/best-epoch=*.ckpt"))
        if ckpt_files:
            best_ckpt = max(ckpt_files, key=lambda x: os.path.getmtime(x))
            best_checkpoints[model_name] = best_ckpt

    if not best_checkpoints:
        print("No trained models found.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Generating animations for {len(best_checkpoints)} models...")

    for model_name, ckpt_path in best_checkpoints.items():
        print(f"\nProcessing {model_name}...")
        
        # Load model
        model = ModelRegistry.build(model_name)
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
        model = lightning_module.model.to(device)
        model.eval()
        
        # Load dataset sample
        is_spatio = "spatiotemporal" in model_name
        cfg.dataset.input_mode = "sequence" if is_spatio else "center_upscale"
        
        try:
            val_dataset = RLPRRestorationDataset.from_config(cfg, project_root=".", split="val")
            sample = val_dataset[0] # Just use the first image
            
            inp = sample["input"].unsqueeze(0).to(device)
            tgt = sample["target"].unsqueeze(0).to(device)
        except Exception as e:
            print(f"Dataset error: {e}. Generating dummy data.")
            channels = 15 if is_spatio else 3
            inp = torch.rand(1, channels, 64, 256).to(device)
            tgt = torch.rand(1, 3, 64, 256).to(device)

        # Setup Hooks to capture activations
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach().cpu()
            return hook
            
        hooks = []
        # Try attaching hooks based on generic layer names across our families
        for name, module in model.named_children():
            if name in ['stem', 'enc1', 'bottleneck', 'dec1', 'head', 'up', 'down', 'encoder_blocks', 'decoder_blocks']:
                hooks.append(module.register_forward_hook(get_activation(name)))

        # Forward pass
        with torch.no_grad():
            pred = model(inp)
            
        for h in hooks: h.remove()
        
        # Calculate metrics
        try:
            psnr_val = compute_psnr(pred.cpu(), tgt.cpu())
            ssim_val = compute_ssim(pred.cpu(), tgt.cpu())
        except:
            psnr_val = 0.0
            ssim_val = 0.0

        # Animate features
        fig, axes = plt.subplots(len(activations) + 2, 1, figsize=(8, 2 * (len(activations) + 2)))
        fig.suptitle(f"{model_name}\nPSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}", fontsize=16)
        
        # Display inputs and outputs statically at the top and bottom
        inp_vis = inp[0, 6:9].cpu().permute(1,2,0) if is_spatio else inp[0].cpu().permute(1,2,0)
        axes[0].imshow(inp_vis.numpy().clip(0,1))
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        axes[-1].imshow(pred[0].cpu().permute(1,2,0).numpy().clip(0,1))
        axes[-1].set_title("Restored Output")
        axes[-1].axis('off')

        # We will animate the channels of the intermediate activations
        # Find the max number of channels among all activations to know how many frames
        max_channels = max([act.shape[1] for act in activations.values()]) if activations else 1
        num_frames = min(max_channels, 60) # Limit frames so it doesn't take forever
        
        im_plots = {}
        layer_names = list(activations.keys())
        
        for i, name in enumerate(layer_names):
            ax = axes[i+1]
            # Initialize with channel 0
            act_tensor = activations[name][0, 0]
            # Normalize for visualization
            act_tensor = (act_tensor - act_tensor.min()) / (act_tensor.max() - act_tensor.min() + 1e-8)
            im_plots[name] = ax.imshow(act_tensor.numpy(), cmap='viridis')
            ax.set_title(f"Layer: {name} | Channel: 0")
            ax.axis('off')

        plt.tight_layout()

        def update(frame):
            for i, name in enumerate(layer_names):
                act = activations[name]
                c = frame % act.shape[1]
                act_tensor = act[0, c]
                act_tensor = (act_tensor - act_tensor.min()) / (act_tensor.max() - act_tensor.min() + 1e-8)
                im_plots[name].set_data(act_tensor.numpy())
                axes[i+1].set_title(f"Layer: {name} | Channel: {c}/{act.shape[1]}")
            return list(im_plots.values())

        print(f"Rendering {num_frames} frames animation for {model_name}...")
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
        
        out_file = out_dir / f"{model_name}_flow.mp4"
        try:
            # Try to save mp4
            ani.save(str(out_file), fps=10)
            print(f"Animation saved to {out_file}")
        except Exception as e:
            print(f"Could not save MP4 (ffmpeg missing?): {e}")
            # Fallback to GIF using Pillow
            gif_file = out_dir / f"{model_name}_flow.gif"
            print("Trying to save as GIF using Pillow instead...")
            try:
                ani.save(str(gif_file), writer='pillow', fps=10)
                print(f"Animation saved to {gif_file}")
            except Exception as e2:
                print(f"Failed to save GIF as well: {e2}")

        plt.close(fig)

if __name__ == "__main__":
    create_animation()
