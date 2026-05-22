"""
Model registry and factory for restoration architecture variants.

Centralized factory for building and tracking all model architectures.
Supports parameter estimation and FLOPs computation for each variant.
"""

from typing import Dict, Type, Optional, Tuple, Any
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a model variant."""

    family: str  # "swinir", "unet", "resnet", "hybrid"
    name: str  # e.g., "swinir_base"
    scale: str  # "small", "base", "large"
    num_params: int
    estimated_flops: Optional[int] = None
    input_channels: int = 3
    output_channels: int = 3
    min_spatial_size: int = 32


class ModelRegistry:
    """Central factory for model variants."""

    _registry: Dict[str, Tuple[Type[nn.Module], Dict]] = {}
    _metadata: Dict[str, ModelMetadata] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[nn.Module], config: Dict, metadata: ModelMetadata):
        """
        Register a model variant.

        Args:
            name: Unique model name (e.g., "swinir_small")
            model_class: Model class to instantiate
            config: Configuration dict for model init
            metadata: ModelMetadata with complexity info
        """
        cls._registry[name] = (model_class, config)
        cls._metadata[name] = metadata
        logger.info(f"[OK] Registered model: {name} ({metadata.num_params / 1e3:.1f}K params)")

    @classmethod
    def build(cls, name: str, **kwargs) -> nn.Module:
        """
        Build a model by name.

        Args:
            name: Model name (must be registered)
            **kwargs: Override config parameters

        Returns:
            Instantiated model
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")

        model_class, base_config = cls._registry[name]
        config = base_config.copy()
        config.update(kwargs)

        model = model_class(**config)
        logger.info(f"Built model: {name}")
        return model

    @classmethod
    def get_metadata(cls, name: str) -> ModelMetadata:
        """Get metadata for a registered model."""
        if name not in cls._metadata:
            raise ValueError(f"Unknown model: {name}")
        return cls._metadata[name]

    @classmethod
    def list_models(cls, family: Optional[str] = None) -> Dict[str, ModelMetadata]:
        """
        List registered models, optionally filtered by family.

        Args:
            family: Optional family name ("swinir", "unet", "resnet", "hybrid")

        Returns:
            Dictionary mapping model name -> metadata
        """
        if family is None:
            return cls._metadata

        return {name: meta for name, meta in cls._metadata.items() if meta.family == family}

    @classmethod
    def print_summary(cls) -> str:
        """Print summary of all registered models."""
        summary_lines = ["Model Registry Summary:", "=" * 80]

        # Group by family
        families = {}
        for name, meta in cls._metadata.items():
            if meta.family not in families:
                families[meta.family] = []
            families[meta.family].append((name, meta))

        for family in sorted(families.keys()):
            summary_lines.append(f"\n{family.upper()}:")
            summary_lines.append("-" * 80)
            summary_lines.append(f"{'Name':<30} {'Scale':<10} {'Params':<15} {'FLOPs':<15}")
            summary_lines.append("-" * 80)

            for name, meta in sorted(families[family]):
                param_str = f"{meta.num_params / 1e6:.2f}M" if meta.num_params >= 1e6 else f"{meta.num_params / 1e3:.1f}K"
                flops_str = (
                    f"{meta.estimated_flops / 1e9:.2f}G"
                    if meta.estimated_flops and meta.estimated_flops >= 1e9
                    else f"{meta.estimated_flops / 1e6:.2f}M"
                    if meta.estimated_flops
                    else "N/A"
                )
                summary_lines.append(f"{name:<30} {meta.scale:<10} {param_str:<15} {flops_str:<15}")

        summary_lines.append("=" * 80)
        return "\n".join(summary_lines)


def estimate_model_flops(model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 256, 256)) -> Optional[int]:
    """
    Estimate FLOPs for a model.

    Uses fvcore if available, otherwise returns None.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape

    Returns:
        Estimated FLOPs or None
    """
    try:
        from fvcore.nn import FlopCountAnalysis

        dummy_input = torch.randn(input_shape)
        flops = FlopCountAnalysis(model, dummy_input).total()
        return int(flops)
    except (ImportError, Exception):
        return None


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Initialize registry with all model variants
# ============================================================================


def initialize_registry():
    """Initialize model registry with all variants."""
    from models.restoration.swinir_unet import SwinIRUNetHybrid
    from models.restoration.unet_variants import UNetLite, UNetStandard, UNetDense
    from models.restoration.resnet_variants import RestorationResNetSmall, RestorationResNetMedium
    from models.restoration.hybrid_attention import HybridAttentionRestoration
    from models.restoration.spatiotemporal_hybrid import SpatiotemporalHybridRestoration

    # SwinIR variants
    # ========================================================================

    # Small (140K params - baseline)
    config_swinir_small = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 32,
        "num_blocks": 4,
        "window_size": 4,
        "num_heads": 4,
        "use_skip_connections": True,
        "gradient_checkpointing": True,
    }
    swinir_small = SwinIRUNetHybrid(**config_swinir_small)
    params_small = count_parameters(swinir_small)
    flops_small = estimate_model_flops(swinir_small)

    ModelRegistry.register(
        "swinir_small",
        SwinIRUNetHybrid,
        config_swinir_small,
        ModelMetadata(
            family="swinir",
            name="swinir_small",
            scale="small",
            num_params=params_small,
            estimated_flops=flops_small,
        ),
    )

    # Base (400K params)
    config_swinir_base = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 48,
        "num_blocks": 5,
        "window_size": 4,
        "num_heads": 4,
        "use_skip_connections": True,
        "gradient_checkpointing": True,
    }
    swinir_base = SwinIRUNetHybrid(**config_swinir_base)
    params_base = count_parameters(swinir_base)
    flops_base = estimate_model_flops(swinir_base)

    ModelRegistry.register(
        "swinir_base",
        SwinIRUNetHybrid,
        config_swinir_base,
        ModelMetadata(
            family="swinir",
            name="swinir_base",
            scale="base",
            num_params=params_base,
            estimated_flops=flops_base,
        ),
    )

    # Large (800K params)
    config_swinir_large = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 64,
        "num_blocks": 6,
        "window_size": 4,
        "num_heads": 4,
        "use_skip_connections": True,
        "gradient_checkpointing": True,
    }
    swinir_large = SwinIRUNetHybrid(**config_swinir_large)
    params_large = count_parameters(swinir_large)
    flops_large = estimate_model_flops(swinir_large)

    ModelRegistry.register(
        "swinir_large",
        SwinIRUNetHybrid,
        config_swinir_large,
        ModelMetadata(
            family="swinir",
            name="swinir_large",
            scale="large",
            num_params=params_large,
            estimated_flops=flops_large,
        ),
    )

    # UNet variants
    # ========================================================================

    # UNet-Lite (180K params)
    config_unet_lite = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 32,
    }
    unet_lite = UNetLite(**config_unet_lite)
    params_unet_lite = count_parameters(unet_lite)
    flops_unet_lite = estimate_model_flops(unet_lite)

    ModelRegistry.register(
        "unet_lite",
        UNetLite,
        config_unet_lite,
        ModelMetadata(
            family="unet",
            name="unet_lite",
            scale="small",
            num_params=params_unet_lite,
            estimated_flops=flops_unet_lite,
        ),
    )

    # UNet-Standard (650K params)
    config_unet_standard = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 48,
    }
    unet_standard = UNetStandard(**config_unet_standard)
    params_unet_standard = count_parameters(unet_standard)
    flops_unet_standard = estimate_model_flops(unet_standard)

    ModelRegistry.register(
        "unet_standard",
        UNetStandard,
        config_unet_standard,
        ModelMetadata(
            family="unet",
            name="unet_standard",
            scale="base",
            num_params=params_unet_standard,
            estimated_flops=flops_unet_standard,
        ),
    )

    # UNet-Dense (1.2M params)
    config_unet_dense = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 64,
    }
    unet_dense = UNetDense(**config_unet_dense)
    params_unet_dense = count_parameters(unet_dense)
    flops_unet_dense = estimate_model_flops(unet_dense)

    ModelRegistry.register(
        "unet_dense",
        UNetDense,
        config_unet_dense,
        ModelMetadata(
            family="unet",
            name="unet_dense",
            scale="large",
            num_params=params_unet_dense,
            estimated_flops=flops_unet_dense,
        ),
    )

    # ResNet variants
    # ========================================================================

    # ResNet-Small (250K params)
    config_resnet_small = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 32,
        "num_blocks": 3,
        "groups": 1,
    }
    resnet_small = RestorationResNetSmall(**config_resnet_small)
    params_resnet_small = count_parameters(resnet_small)
    flops_resnet_small = estimate_model_flops(resnet_small)

    ModelRegistry.register(
        "resnet_small",
        RestorationResNetSmall,
        config_resnet_small,
        ModelMetadata(
            family="resnet",
            name="resnet_small",
            scale="small",
            num_params=params_resnet_small,
            estimated_flops=flops_resnet_small,
        ),
    )

    # ResNet-Medium (750K params)
    config_resnet_medium = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 48,
        "num_levels": 3,
        "groups": 2,
    }
    resnet_medium = RestorationResNetMedium(**config_resnet_medium)
    params_resnet_medium = count_parameters(resnet_medium)
    flops_resnet_medium = estimate_model_flops(resnet_medium)

    ModelRegistry.register(
        "resnet_medium",
        RestorationResNetMedium,
        config_resnet_medium,
        ModelMetadata(
            family="resnet",
            name="resnet_medium",
            scale="medium",
            num_params=params_resnet_medium,
            estimated_flops=flops_resnet_medium,
        ),
    )

    # Hybrid Attention variant
    # ========================================================================

    # Hybrid Attention (550K params)
    config_hybrid_attention = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 56,
        "num_blocks": 4,
        "window_size": 4,
        "num_heads": 4,
    }
    hybrid_att = HybridAttentionRestoration(**config_hybrid_attention)
    params_hybrid = count_parameters(hybrid_att)
    flops_hybrid = estimate_model_flops(hybrid_att)

    ModelRegistry.register(
        "hybrid_attention",
        HybridAttentionRestoration,
        config_hybrid_attention,
        ModelMetadata(
            family="hybrid",
            name="hybrid_attention",
            scale="base",
            num_params=params_hybrid,
            estimated_flops=flops_hybrid,
        ),
    )

    # Spatiotemporal Hybrid Attention
    config_st_hybrid_base = {
        "in_channels": 15,  # 5 frames
        "out_channels": 3,
        "base_channels": 56,
        "num_blocks": 4,
        "window_size": 4,
        "num_heads": 4,
    }
    st_hybrid_base = SpatiotemporalHybridRestoration(**config_st_hybrid_base)
    params_st_hybrid_base = count_parameters(st_hybrid_base)
    flops_st_hybrid_base = estimate_model_flops(st_hybrid_base)

    ModelRegistry.register(
        "spatiotemporal_hybrid_base",
        SpatiotemporalHybridRestoration,
        config_st_hybrid_base,
        ModelMetadata(
            family="hybrid",
            name="spatiotemporal_hybrid_base",
            scale="base",
            num_params=params_st_hybrid_base,
            estimated_flops=flops_st_hybrid_base,
            input_channels=15,
        ),
    )

    config_st_hybrid_small = {
        "in_channels": 15,
        "out_channels": 3,
        "base_channels": 32,
        "num_blocks": 3,
        "window_size": 4,
        "num_heads": 4,
    }
    st_hybrid_small = SpatiotemporalHybridRestoration(**config_st_hybrid_small)
    params_st_hybrid_small = count_parameters(st_hybrid_small)
    flops_st_hybrid_small = estimate_model_flops(st_hybrid_small)

    ModelRegistry.register(
        "spatiotemporal_hybrid_small",
        SpatiotemporalHybridRestoration,
        config_st_hybrid_small,
        ModelMetadata(
            family="hybrid",
            name="spatiotemporal_hybrid_small",
            scale="small",
            num_params=params_st_hybrid_small,
            estimated_flops=flops_st_hybrid_small,
            input_channels=15,
        ),
    )


if __name__ == "__main__":
    # Initialize and print registry
    initialize_registry()
    print(ModelRegistry.print_summary())

    # Test building a model
    model = ModelRegistry.build("swinir_small")
    print(f"\nBuilt model: {model}")
    print(f"Trainable params: {count_parameters(model):,}")
