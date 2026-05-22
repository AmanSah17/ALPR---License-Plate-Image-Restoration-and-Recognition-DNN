import argparse
import logging
from pathlib import Path
import torch
from tabulate import tabulate

from datasets.rlpr_dataset import RLPRDataset
from engine.inference_pipeline import ALPRPipeline
from utils.checkpoint_utils import resolve_best_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def compute_cer(pred: str, gt: str) -> float:
    import Levenshtein
    pred = pred.lower().replace(" ", "")
    gt = gt.lower().replace(" ", "")
    if len(gt) == 0:
        return 0.0
    return Levenshtein.distance(pred, gt) / len(gt)

def find_best_checkpoint(model_name: str) -> Path:
    return resolve_best_checkpoint(model_name, prefer_metric="psnr")

def main():
    parser = argparse.ArgumentParser(description="End-to-End ALPR Pipeline Inference")
    parser.add_argument("--model", type=str, default="unet_standard", help="Model family to run (e.g. swinir_base, unet_standard, spatiotemporal_hybrid_large)")
    parser.add_argument("--ckpt", type=str, default=None, help="Explicit path to checkpoint")
    parser.add_argument("--ocr-backend", type=str, default="fast_plate_ocr", help="OCR backend to use at inference time")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to process from dataset")
    parser.add_argument("--dataset-path", type=str, default="Realistic License Plate Restoration and Recognition Dataset (RLPR)", help="Path to RLPR dataset")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Resolve checkpoint
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        logger.info(f"Dynamically locating best checkpoint for '{args.model}'...")
        ckpt_path = find_best_checkpoint(args.model)
        
    # 2. Initialize pipeline
    pipeline = ALPRPipeline(
        model_name=args.model,
        checkpoint_path=str(ckpt_path),
        ocr_backend=args.ocr_backend,
        device=device,
    )
    
    # 3. Load Dataset
    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = RLPRDataset(root_dir=args.dataset_path, num_frames=31)
    
    # 4. Inference Loop
    results = []
    
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        
        # Prepare input tensor based on model requirement
        if "spatiotemporal" in args.model:
            # Requires T=5 sequence. Get center frames
            center = sample["center_frame_index"]
            frames = sample["frames"] # (T, C, H, W)
            # Take 5 frames around center: center-2 to center+2
            start = max(0, center - 2)
            end = min(frames.shape[0], center + 3)
            seq = frames[start:end]
            # Reshape (5, 3, H, W) to (15, H, W)
            inp = seq.view(-1, seq.shape[2], seq.shape[3]).unsqueeze(0)
        else:
            # Single center frame
            inp = sample["center_frame"].unsqueeze(0)
            
        ground_truth = sample["plate_text_compact"]
        
        # Resize input to match pseudo_gt_roi shape (as done during training)
        target_shape = sample["pseudo_gt_roi"].shape[1:] # (H, W)
        import torch.nn.functional as F
        inp = F.interpolate(inp, size=target_shape, mode="bilinear", align_corners=False)
        
        # Run pipeline
        out = pipeline(inp)
        
        # Save images for debug
        import torchvision
        torchvision.utils.save_image(out['restored'], f"debug_{i}_restored.png")
        torchvision.utils.save_image(out['refined'], f"debug_{i}_refined.png")
        torchvision.utils.save_image(sample["pseudo_gt_roi"], f"debug_{i}_gt.png")
        
        pred_text = out['text'][0]
        conf = float(out['confidence'][0]) if out['confidence'] else 0.0
        
        cer = compute_cer(pred_text, ground_truth)
        
        results.append([
            sample["sample_id"],
            ground_truth,
            pred_text,
            f"{conf:.2f}",
            f"{cer:.3f}"
        ])
        
    # Print results
    print("\n" + "="*80)
    print(f"End-to-End Pipeline Results ({args.model})")
    print("="*80)
    headers = ["Sample ID", "Ground Truth", "Predicted", "Confidence", "CER"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    avg_cer = sum(float(r[4]) for r in results) / len(results) if results else 0
    print(f"\nAverage CER: {avg_cer:.3f}")
    print("="*80)

if __name__ == "__main__":
    main()
