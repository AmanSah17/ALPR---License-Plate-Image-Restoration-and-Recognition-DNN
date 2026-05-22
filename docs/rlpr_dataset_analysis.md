# RLPR Dataset Analysis

## Scope

This note summarizes the local RLPR release stored in:

- `D:\ALPR_Research\Realistic License Plate Restoration and Recognition Dataset (RLPR)`

It combines:

- direct inspection of the downloaded dataset files
- reverse engineering of the bundled utility scripts
- cross-checking with the MF-LPR paper and the RLPR dataset release description

## Primary Takeaways

- RLPR is an evaluation-focused multi-frame license plate restoration benchmark, not a training-scale dataset.
- Each sample is a short sequence of `31` cropped low-quality license plate frames.
- The reference frame for alignment is the temporal center frame, which is `Plate_crop/16.png`.
- Every sample provides a pseudo ground-truth plate image aligned to the center low-quality frame.
- Labels are stored only by line order in `Label/Labels.txt`; there is no explicit `sample_id -> label` mapping file.
- Bounding boxes and pseudo ground-truth alignment were manually refined, so mild size and position variation is part of the dataset by design.

## What The Paper Says

The MF-LPR paper describes RLPR as a realistic benchmark built from real dash-cam footage rather than synthetically degraded images.

Operationally relevant details from the paper:

- The authors started from `1,052` dash-cam clips and retained `200` sequences after manual screening.
- Each retained sample contains `31` consecutive low-quality frames.
- The pseudo ground truth was extracted from a higher-quality frame in the same clip and manually aligned to the center low-quality frame with a homography.
- Plate regions were first detected with a fine-tuned DeepLabV3 model and then manually refined.
- The paper explicitly says RLPR is valuable as an evaluation dataset, but not suitable as a primary training dataset because of the limited sample count.

Paper and dataset references:

- MF-LPR paper: `doi:10.1016/j.cviu.2025.104361`
- RLPR dataset release: `doi:10.17632/4rs5wpvckz.2`

## Local Release Structure

The local release has three top-level directories:

- `Dataset/`
- `Label/`
- `Utility/`

Each `Dataset/sample_XXX/` directory contains:

- `Plate_crop/01.png` through `Plate_crop/31.png`
- `Pseudo_GT.png`
- `Pseudo_GT_ROI.png`
- `SR_ROI.png`
- `Select_ROI/coordinate/Roi_coordinate.txt`
- `Homography_Transformation/coordinate/HP_coordinate.txt`
- `Homography_Transformation/coordinate/LP_coordinate.txt`

Observed semantics:

- `Plate_crop/*.png`: low-quality cropped plate sequence used as model input.
- `16.png`: center frame and natural anchor frame for flow estimation, warping, and temporal fusion.
- `Pseudo_GT.png`: manually aligned pseudo ground-truth image in the pseudo-GT source frame resolution.
- `Pseudo_GT_ROI.png`: cropped pseudo-GT region intended for direct comparison against restored output.
- `SR_ROI.png`: the release includes a restoration-sized ROI image with the same dimensions as `Pseudo_GT_ROI.png` for every sample.
- `Roi_coordinate.txt`: selected ROI coordinates used to define the final comparable region.
- `HP_coordinate.txt` and `LP_coordinate.txt`: manually clicked point correspondences used for homography alignment.

## What The Bundled Utility Code Confirms

The `Utility/` folder is important because it clarifies how pseudo ground truth was built.

Confirmed behavior from the script:

- a higher-quality plate image is paired with the center low-quality frame `16.png`
- four corner points are manually selected in both images
- a homography is estimated from those point pairs
- the higher-quality image is warped into the low-quality frame geometry

This matters because it tells us the ground truth is:

- pseudo aligned, not natively co-registered
- sample-specific, not produced by a uniform synthetic degradation pipeline
- appropriate for evaluation, but not something we should treat as perfectly rigid or artifact-free supervision

## Local Audit Results

The local dataset audit found:

- `200` sample directories
- `200` label lines
- `31` low-quality frames in every sample
- no missing required files in the checked release
- no `Pseudo_GT_ROI` vs `SR_ROI` size mismatches

Sequence and label statistics:

- total label characters: `1261`
- label length distribution: `139` samples with `6` digits, `61` samples with `7` digits
- one label line is stored without an internal space: `469399`

Important loader implication:

- labels must be treated as raw strings
- spaces should be preserved for formatting metadata if needed
- OCR targets should usually be normalized with `label.replace(" ", "")` for character-level evaluation

### Input Frame Size Statistics

For the center low-quality frame (`16.png`):

- width: min `68`, mean `85.28`, median `80`, max `148`
- height: min `17`, mean `21.32`, median `20`, max `37`

This means RLPR does not use a fixed crop size. The training and evaluation pipeline will need:

- dynamic resizing or padding
- aspect-ratio-aware batching
- careful handling of very small plate crops

### Pseudo-GT ROI Size Statistics

For `Pseudo_GT_ROI.png`:

- width: min `91`, mean `215.35`, median `211`, max `393`
- height: min `24`, mean `54.22`, median `52`, max `121`

This also confirms that the pseudo-GT comparison target is variable-sized.

## Scale Factor Reality Check

One tempting assumption is that RLPR is a simple `x4` super-resolution dataset. The local release shows that this is only mostly true for `Pseudo_GT.png`, not universally true for the final ROI target.

Observed facts:

- `184 / 200` samples have `Pseudo_GT.png` exactly `4x` the center frame dimensions
- `16 / 200` samples do not follow exact `4x` scaling
- `Pseudo_GT_ROI.png` size ratios are variable, with average width and height scale near `2.57x`

Engineering implication:

- do not hardcode RLPR as a pure fixed-scale SR dataset
- design the restoration module and losses to support variable target sizes
- treat the task as multi-frame restoration plus recognition, not only classic super-resolution

## Sample Contract For A Robust Loader

For our framework, a single sample should expose at least:

- `sample_id`
- `frames`: ordered list of `31` images
- `center_frame_index`: `15` in zero-based indexing
- `center_frame_path`
- `pseudo_gt_roi`
- `pseudo_gt_full`
- `sr_roi`
- `plate_text_raw`
- `plate_text_compact`
- `roi_coordinates`
- `homography_high_quality_points`
- `homography_low_quality_points`

Recommended indexing rule:

- map `sample_001` to line `1` of `Labels.txt`
- map `sample_200` to line `200` of `Labels.txt`

Because there is no manifest file, the loader should validate that:

- sample directories are contiguous and sortable
- label count equals sample count
- frame names are exactly `01.png` to `31.png`

## Research Implications For MF-LPR2-Style Development

RLPR strongly shapes the framework design:

- Since only `200` samples exist, RLPR should be treated as a benchmark and final evaluation dataset first.
- We will likely need transfer learning, synthetic pretraining, or self-supervised warm starts before RLPR fine-tuning.
- Temporal alignment should be center-frame-centric because the pseudo-GT is aligned to frame `16`.
- Evaluation masks or ROI-aware losses are important because the GT is ROI-oriented rather than a dense scene image target.
- OCR metrics should be first-class citizens because the dataset exists for restoration plus recognition, not just visual enhancement.
- Artifact-sensitive evaluation matters because the paper explicitly motivates avoiding hallucinated characters.

## Immediate Phase-2 Design Constraints

When we implement `RLPRDataset`, it should:

- load all `31` frames in deterministic order
- expose both raw and normalized OCR labels
- support variable-size samples safely
- allow optional use of `Pseudo_GT_ROI` and `SR_ROI`
- validate per-sample completeness at initialization time
- expose metadata needed for future PDNF-style evaluation

Recommended defaults for the first dataset implementation:

- use `16.png` as the restoration anchor
- keep geometric transforms synchronized across all `31` frames and the target
- keep photometric augmentation conservative to avoid destroying tiny characters
- perform resize and padding in a config-driven collate function rather than hardcoding image shape into the dataset class

## Recommended Next Step

The next implementation step should be Phase-1 plus a thin Phase-2 foundation:

- configuration system
- logging and artifact directories
- deterministic utilities
- RLPR dataset validator and manifest builder

That order keeps the dataset loader reproducible from the start and avoids reworking paths, logging, and experiment metadata later.
