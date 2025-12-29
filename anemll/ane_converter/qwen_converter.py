"""Converter for Qwen 3 models.

This module provides a lightweight converter that mirrors the
:class:`LlamaConverter` behaviour for Qwen models without inheriting from
it. Only the pieces required for the unit tests are implemented."""

from __future__ import annotations

import argparse
import os
import warnings
try:
    from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
except Exception:  # pragma: no cover - sklearn optional
    SklearnConvergenceWarning = None

# Globally suppress the sklearn ConvergenceWarning (applies to multiprocessing workers too)
if SklearnConvergenceWarning is not None:
    warnings.filterwarnings("ignore", category=SklearnConvergenceWarning)
warnings.filterwarnings("ignore", message="Number of distinct clusters .* smaller than n_clusters")
from typing import Optional, List

import numpy as np
import torch
import coremltools as ct
import coremltools.optimize as cto

from .environment import require_coreml

from .base_converter import BaseConverter
from .metadata import AddMetadata, ModelPart
from ..models.qwen_model import (
    QwenForCausalLM,
    QwenConfig,
    MODEL_DTYPE,
    TEST_DEVICE,
    CONTEXT_LENGTH,
)
from ..models.aq1_mil_layer import (
    build_aq1_conv_layer,
    load_aq1_checkpoint,
    get_layer_aq1_data,
    compute_aq1_storage_size,
)
from ..models.anemll_quant import AnemllConv2d


# =============================================================================
# A1F: Constant Folding Bypass for Factored Scales
# =============================================================================
# These helpers prevent CoreML from folding const*const during conversion,
# allowing snapped weights and factored scales (A @ B) to remain separate.
# See tests/dev/A1F_FOLDING_ISSUE.md for detailed documentation.

def _patch_value_inference(threshold: int = 100000):
    """
    Patch value_inference for both elementwise_binary AND matmul to skip
    evaluation for large tensors.

    This prevents CoreML from folding const*const and const@const, keeping
    snapped weights and factored scales (A @ B) as separate operations.

    Args:
        threshold: Skip evaluation if tensor size > threshold (default 100K elements)

    Returns:
        Tuple of (original_binary, original_matmul) for restoration
    """
    from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_binary import elementwise_binary
    from coremltools.converters.mil.mil.ops.defs.iOS15.linear import matmul as matmul_op

    original_binary = elementwise_binary.value_inference
    original_matmul = matmul_op.value_inference

    def patched_binary_value_inference(self):
        try:
            x_shape = self.x.shape if hasattr(self.x, 'shape') else ()
            y_shape = self.y.shape if hasattr(self.y, 'shape') else ()

            try:
                x_size = int(np.prod([int(s) for s in x_shape]))
            except (TypeError, ValueError):
                x_size = 0  # Symbolic shape, allow evaluation

            try:
                y_size = int(np.prod([int(s) for s in y_shape]))
            except (TypeError, ValueError):
                y_size = 0  # Symbolic shape, allow evaluation

            if x_size > threshold or y_size > threshold:
                return None  # Skip evaluation for large tensors

        except Exception:
            pass  # Fall through to original on any error

        return original_binary(self)

    def patched_matmul_value_inference(self):
        """Skip matmul evaluation for large matrices (scale_A @ scale_B)."""
        try:
            x_shape = self.x.shape if hasattr(self.x, 'shape') else ()
            y_shape = self.y.shape if hasattr(self.y, 'shape') else ()

            try:
                x_size = int(np.prod([int(s) for s in x_shape]))
            except (TypeError, ValueError):
                x_size = 0

            try:
                y_size = int(np.prod([int(s) for s in y_shape]))
            except (TypeError, ValueError):
                y_size = 0

            # For matmul, use lower threshold since scale matrices are smaller
            # but we still want to prevent A @ B folding
            matmul_threshold = min(threshold, 10000)  # 10K elements for matmul
            if x_size > matmul_threshold or y_size > matmul_threshold:
                return None  # Skip evaluation for scale matrices

        except Exception:
            pass

        return original_matmul(self)

    elementwise_binary.value_inference = patched_binary_value_inference
    matmul_op.value_inference = patched_matmul_value_inference
    print(f"[A1F] Patched value_inference (binary + matmul) to skip tensors > {threshold} elements")
    return (original_binary, original_matmul)


def _restore_value_inference(originals):
    """Restore original value_inference after conversion."""
    from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_binary import elementwise_binary
    from coremltools.converters.mil.mil.ops.defs.iOS15.linear import matmul as matmul_op

    original_binary, original_matmul = originals
    elementwise_binary.value_inference = original_binary
    matmul_op.value_inference = original_matmul
    print("[A1F] Restored original value_inference (binary + matmul)")


def _apply_selective_palettization(mlmodel, snapped_op_names: list, mode: str = "unique"):
    """
    Apply palettization only to snapped weights, preserving A @ B structure.

    Standard palettize_weights() runs const_elimination which folds matmul(A, B).
    This function uses _apply_graph_pass + manual mil_convert to avoid that.

    Args:
        mlmodel: CoreML model to palettize
        snapped_op_names: List of const op names for snapped weights
        mode: Palettization mode ("unique" for exact values, "kmeans" for clustering)

    Returns:
        Palettized CoreML model with A @ B preserved
    """
    import coremltools.optimize.coreml as cto_coreml
    from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
    from coremltools.converters.mil.mil.passes.graph_pass import PassOption
    from coremltools.models.utils import _apply_graph_pass
    from coremltools.converters.mil.converter import mil_convert as _mil_convert

    print(f"[A1F] Palettizing {len(snapped_op_names)} snapped weights with mode='{mode}'...")

    # Configure selective palettization
    op_name_configs = {}
    for op_name in snapped_op_names:
        op_name_configs[op_name] = cto_coreml.OpPalettizerConfig(
            mode=mode,
            granularity="per_tensor"
        )

    config = cto_coreml.OptimizationConfig(
        global_config=None,  # Don't palettize by default
        op_name_configs=op_name_configs
    )

    # Get palettization pass from registry
    weight_palettizer = PASS_REGISTRY["compression::palettize_weights"]
    weight_palettizer.set_options([
        PassOption("config", config),
        PassOption("joint_compression", False)
    ])

    # Apply pass but return pymil program (before _mil_convert runs const_elimination)
    palettized_prog = _apply_graph_pass(mlmodel, weight_palettizer, return_pymil_prog=True)

    # Convert with custom pipeline (NO const_elimination)
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::const_elimination"])

    spec = mlmodel.get_spec()
    palettized = _mil_convert(
        palettized_prog,
        convert_to="mlprogram",
        convert_from="milinternal",
        specification_version=spec.specificationVersion,
        compute_units=mlmodel.compute_unit,
        model_description=spec.description,
        skip_model_load=False,
        pass_pipeline=pipeline,
    )

    print(f"[A1F] Palettization complete")
    return palettized


def _find_snapped_ops(mlmodel) -> list:
    """Find snapped weight const ops in the MIL program."""
    snapped_ops = []
    mil_prog = mlmodel._mil_program

    if not mil_prog:
        return snapped_ops

    for func in mil_prog.functions.values():
        for op in func.operations:
            if op.op_type == 'const':
                try:
                    val = op.val.val if hasattr(op.val, 'val') else None
                    if val is not None and hasattr(val, 'size') and val.size > 1000:
                        name = op.outputs[0].name if op.outputs else op.name
                        if 'snapped' in name:
                            snapped_ops.append(name)
                except Exception:
                    pass

    return snapped_ops


class QwenConverter(BaseConverter):
    """Handle conversion of Qwen 3 models to Core ML."""

    model_cls = QwenForCausalLM

    def __init__(
        self,
        model: QwenForCausalLM,
        context_length: int = CONTEXT_LENGTH,
        batch_size: int = 64,
        lut_bits: int | None = 4,
        per_channel: int = 8,
        num_chunks: int = 1,
        aq1_checkpoint: Optional[str] = None,
        aq1_nbits_mlp: int = 2,
        aq1_nbits_attn: int = 4,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__(model)
        self.context_length = context_length
        self.batch_size = batch_size
        self.lut_bits = lut_bits
        self.per_channel = per_channel
        self.head_dim = (
            model.model.config.hidden_size // model.model.config.num_attention_heads
        )
        self.converted_model = None
        self.num_chunks = num_chunks
        # AQ1 settings
        self.aq1_checkpoint = aq1_checkpoint
        self.aq1_nbits_mlp = aq1_nbits_mlp
        self.aq1_nbits_attn = aq1_nbits_attn
        self.aq1_data = None  # Loaded on demand
        self.num_layers = num_layers  # For limiting layers in AQ1 conversion

    @staticmethod
    def GetTransformerStates(model, part=None, prefix="model.model."):
        """Get the transformer states for CoreML conversion"""
        head_dim = getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        )
        num_layers = (
            model.config.num_hidden_layers
        )  # Get total number of layers from config

        # For unified cache
        num_layers_this_part = num_layers * 2
        print(
            f"GetTransformerStates part={part} num_layers_this_part={num_layers_this_part} model.config.num_hidden_layers={model.config.num_hidden_layers}"
        )
        print(f"Using head_dim={head_dim} from config")

        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(
                        num_layers_this_part,
                        model.config.num_key_value_heads,
                        model.config.state_length,
                        head_dim,
                    ),
                    dtype=np.float16,
                ),
                name=f"{prefix}kv_cache_0",  # Only one group for unified cache
            )
        ]
        return states

    def postprocess(self, num_workers=None):
        """Apply LUT quantization if configured.

        Args:
            num_workers: Optional number of workers for parallel processing.
                        If None, uses default single worker.
        """
        if self.converted_model is not None and self.lut_bits is not None:
            print(
                f"Applying LUT quantization with {self.lut_bits} bits and {self.per_channel} channels per group using {num_workers if num_workers else 1} worker(s)..."
            )
            try:
                # Suppress sklearn ConvergenceWarning during quantization
                with warnings.catch_warnings():
                    if SklearnConvergenceWarning is not None:
                        warnings.simplefilter('ignore', SklearnConvergenceWarning)
                    warnings.simplefilter('ignore', UserWarning)
                    # Set up quantization config
                    config = cto.coreml.OptimizationConfig(
                        global_config=cto.coreml.OpPalettizerConfig(
                            mode="kmeans",
                            nbits=self.lut_bits,
                            granularity="per_grouped_channel",
                            group_size=self.per_channel,
                            num_kmeans_workers=(
                                num_workers if num_workers is not None else 1
                            ),  # Use provided workers or default to 1
                        ),
                    )

                    # Apply quantization
                    self.converted_model = cto.coreml.palettize_weights(
                        self.converted_model, config
                    )
                print("✅ LUT quantization completed successfully")

            except Exception as e:
                print(f"❌ LUT quantization failed: {str(e)}")
                print("Continuing without quantization...")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(
        self, part: str = "full"
    ) -> ct.models.MLModel | List[ct.models.MLModel]:
        """Convert the wrapped model to CoreML format.

        Args:
            part: Which part of the model to convert:
                 "full" - complete model (default)
                 "prefill" - prefill mode for initial sequence processing
                 "embeddings" - embeddings only (input_ids -> hidden_states)

        Returns:
            ct.models.MLModel: Converted model
        """
        print(f"QwenConverter.convert() called with part={part}")
        require_coreml()
        print("Calling preprocess()...")
        self.preprocess()

        if part in ("full", "all", "123"):
            print("Converting full model...")
            mlmodel = self.convert_to_coreml(self.model)
        elif part == "monolithic":
            print("Converting monolithic model (embeddings + FFN + LM head)...")
            mlmodel = self.convert_monolithic(self.model, is_prefill=False)
        elif part == "monolithic_prefill":
            print("Converting monolithic prefill model...")
            mlmodel = self.convert_monolithic(self.model, is_prefill=True)
        elif part in ("embeddings", "1"):
            print("Converting embeddings...")
            mlmodel = self.convert_part_1(self.model)
        elif part in ("prefill", "2_prefill"):
            print("Converting prefill...")
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2_prefill(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2_prefill(self.model)
        elif part == "2":
            print("Converting FFN...")
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2(self.model)
        elif part == "aq1_mlp":
            print("Converting FFN with AQ1 MIL (constexpr_lut_to_dense + dynamic A*B)...")
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2_aq1_mlp(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2_aq1_mlp(self.model)
        elif part == "aq1_ffn":
            print("Converting FFN with AnemllConv2d.build_mil_conv (constexpr + dynamic A*B)...")
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2_aq1_conv(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2_aq1_conv(self.model)
        elif part == "aq1_full":
            print("Converting full transformer with AQ1 (attention + KV cache + MLP)...")
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2_aq1_full(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2_aq1_full(self.model)
        elif part == "a1f":
            print("Converting FFN with A1F approach (LUT + factored scales)...")
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2_a1f(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2_a1f(self.model)
        elif part == "a1f_prefill":
            print("Converting prefill with A1F approach (LUT + factored scales)...")
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2_a1f_prefill(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2_a1f_prefill(self.model)
        elif part == "3":
            print("Converting LM head...")
            mlmodel = self.convert_part_3(self.model)
        else:
            raise ValueError(f"Unsupported part: {part}")

        print("Calling postprocess()...")
        self.postprocess()
        print("QwenConverter.convert() completed")
        return mlmodel

    def convert_to_coreml(self, model: QwenForCausalLM) -> ct.models.MLModel:
        """Convert the entire model to CoreML."""
        require_coreml()
        print("Creating wrapper model...")

        class Wrapper(torch.nn.Module):
            def __init__(self, model: QwenForCausalLM, context_length: int) -> None:
                super().__init__()
                self.model = model
                self.context_length = context_length

            def forward(
                self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                causal_mask: torch.Tensor,
                current_pos: torch.Tensor,
                update_mask: torch.Tensor,
            ) -> torch.Tensor:
                # Fixed window approach: return full logits, extract position on Python side
                return self.model(
                    input_ids=input_ids,
                    update_mask=update_mask,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    IN_PREFILL=False,
                )

        wrapper = Wrapper(model, self.context_length)
        wrapper.eval()
        print("Wrapper model created and set to eval mode")

        print("Preparing model inputs for tracing...")
        # Use single token approach for KV cache compatibility
        sample_input_ids = torch.zeros(
            (1, 1), dtype=torch.int32, device=TEST_DEVICE
        )  # [1, 1] - single token
        sample_position_ids = torch.zeros(
            (1,), dtype=torch.int32, device=TEST_DEVICE
        )  # [1] - single position
        sample_causal_mask = torch.zeros(
            (1, 1, 1, self.context_length), dtype=torch.float16, device=TEST_DEVICE
        )  # [1, 1, 1, context_length]
        sample_current_pos = torch.zeros(
            (1,), dtype=torch.int32, device=TEST_DEVICE
        )  # [1] - current position
        sample_update_mask = torch.zeros(
            (1, 1, self.context_length, 1), dtype=torch.float16, device=TEST_DEVICE
        )  # [1, 1, context_length, 1]
        print("Sample inputs created (Single Token)")
        print(f"sample_input_ids shape: {sample_input_ids.shape}")
        print(f"sample_position_ids shape: {sample_position_ids.shape}")
        print(f"sample_causal_mask shape: {sample_causal_mask.shape}")
        print(f"sample_current_pos shape: {sample_current_pos.shape}")
        print(f"sample_update_mask shape: {sample_update_mask.shape}")

        print("Starting torch.jit.trace...")
        traced = torch.jit.trace(
            wrapper,
            (
                sample_input_ids,
                sample_position_ids,
                sample_causal_mask,
                sample_current_pos,
                sample_update_mask,
            ),
        )
        print("torch.jit.trace completed!")

        # Register CoreML converter for custom ops if dynamic mode is enabled
        use_dynamic = os.environ.get('ANEMLL_DYNAMIC_SCALES', '0') == '1'
        if use_dynamic:
            from ..models.anemll_quant import ensure_coreml_converter_registered
            ensure_coreml_converter_registered()
            print("  Registered CoreML converter for anemll_quant::quant_conv")

        print("Starting CoreML conversion...")

        # Dynamic mode: patch value_inference to prevent A@B folding
        if use_dynamic:
            original_value_inference = _patch_value_inference(threshold=100000)
            print("  Applied value_inference patch to prevent const folding")

        try:
            mlmodel = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(
                        name="input_ids", shape=sample_input_ids.shape, dtype=np.int32
                    ),
                    ct.TensorType(
                        name="position_ids", shape=sample_position_ids.shape, dtype=np.int32
                    ),
                    ct.TensorType(
                        name="causal_mask", shape=sample_causal_mask.shape, dtype=np.float16
                    ),
                    ct.TensorType(
                        name="current_pos", shape=sample_current_pos.shape, dtype=np.int32
                    ),
                    ct.TensorType(
                        name="update_mask", shape=sample_update_mask.shape, dtype=np.float16
                    ),
                ],
                outputs=[
                    ct.TensorType(name="logits1", dtype=np.float16),
                    ct.TensorType(name="logits2", dtype=np.float16),
                    ct.TensorType(name="logits3", dtype=np.float16),
                    ct.TensorType(name="logits4", dtype=np.float16),
                    ct.TensorType(name="logits5", dtype=np.float16),
                    ct.TensorType(name="logits6", dtype=np.float16),
                    ct.TensorType(name="logits7", dtype=np.float16),
                    ct.TensorType(name="logits8", dtype=np.float16),
                    ct.TensorType(name="logits9", dtype=np.float16),
                    ct.TensorType(name="logits10", dtype=np.float16),
                    ct.TensorType(name="logits11", dtype=np.float16),
                    ct.TensorType(name="logits12", dtype=np.float16),
                    ct.TensorType(name="logits13", dtype=np.float16),
                    ct.TensorType(name="logits14", dtype=np.float16),
                    ct.TensorType(name="logits15", dtype=np.float16),
                    ct.TensorType(name="logits16", dtype=np.float16),
                ],
                states=self.GetTransformerStates(model, part=None, prefix="model.model."),
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.iOS18,
                convert_to="mlprogram",
            )
        finally:
            if use_dynamic:
                _restore_value_inference(original_value_inference)
                print("  Restored value_inference")

        print("CoreML conversion completed!")

        # Apply LUT quantization if specified
        if self.lut_bits:
            self.converted_model = mlmodel  # Set for postprocess
            self.postprocess(num_workers=8)  # Allow passing num_workers if needed
            mlmodel = self.converted_model

        return mlmodel

    # --------------------------------------------------------------
    # Part-based conversion helpers
    # --------------------------------------------------------------
    def convert_part_1(self, model: QwenForCausalLM) -> ct.models.MLModel:
        """Convert embeddings layer only."""
        require_coreml()
        return self.convert_embeddings(model)

    def convert_part_3(self, model: QwenForCausalLM) -> ct.models.MLModel:
        """Convert LM head only."""
        require_coreml()

        class LMHeadWrapper(torch.nn.Module):
            def __init__(self, model: QwenForCausalLM) -> None:
                super().__init__()
                if hasattr(model, "lm_head16_1"):
                    self.heads = [
                        getattr(model, f"lm_head16_{i}") for i in range(1, 17)
                    ]
                    self.mode = "16"
                elif hasattr(model, "lm_head8_1"):
                    self.heads = [getattr(model, f"lm_head8_{i}") for i in range(1, 9)]
                    self.mode = "8"
                elif hasattr(model, "lm_head2_1"):
                    self.heads = [model.lm_head2_1, model.lm_head2_2]
                    self.mode = "2"
                elif hasattr(model, "lm_head1"):
                    self.head = model.lm_head1
                    self.mode = "1"
                else:
                    self.head = model.lm_head
                    self.mode = "linear"

            def forward(self, hidden_states: torch.Tensor):
                if self.mode != "linear":
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)

                if self.mode == "16":
                    return tuple(
                        h(hidden_states).squeeze(2).transpose(1, 2) for h in self.heads
                    )
                if self.mode == "8":
                    return tuple(
                        h(hidden_states).squeeze(2).transpose(1, 2) for h in self.heads
                    )
                if self.mode == "2":
                    logits1 = self.heads[0](hidden_states).squeeze(2).transpose(1, 2)
                    logits2 = self.heads[1](hidden_states).squeeze(2).transpose(1, 2)
                    return logits1, logits2
                if self.mode == "1":
                    return self.head(hidden_states).squeeze(2).transpose(1, 2)
                return self.head(hidden_states)

        wrapper = LMHeadWrapper(model)
        wrapper.eval()
        
        # Ensure no gradients
        for param in wrapper.parameters():
            param.requires_grad = False

        sample_input = torch.zeros(
            (1, 1, model.config.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE
        )
        
        # Trace with no_grad context
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, sample_input)

        if getattr(wrapper, "mode") == "16":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16) for i in range(1, 17)
            ]
        elif getattr(wrapper, "mode") == "8":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16) for i in range(1, 9)
            ]
        elif getattr(wrapper, "mode") == "2":
            outputs = [
                ct.TensorType(name="logits1", dtype=np.float16),
                ct.TensorType(name="logits2", dtype=np.float16),
            ]
        else:
            outputs = [ct.TensorType(name="logits", dtype=np.float16)]

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="hidden_states", shape=sample_input.shape, dtype=np.float16
                )
            ],
            outputs=outputs,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)
            mlmodel = self.converted_model

        return mlmodel

    def convert_part_2(
        self, model: QwenForCausalLM, chunk_idx: int = 0, total_chunks: int = 1
    ) -> ct.models.MLModel:
        """Convert transformer layers for generation (FFN)."""
        require_coreml()

        # Check if dynamic mode is enabled
        use_dynamic = os.environ.get('ANEMLL_DYNAMIC_SCALES', '0') == '1'
        if use_dynamic:
            from ..models.anemll_quant import ensure_coreml_converter_registered
            ensure_coreml_converter_registered()
            print("  Dynamic mode: registered CoreML converter for constexpr_lut_to_dense + matmul")

        # Load AQ1 checkpoint if provided - replaces Conv2d with AnemllConv2d
        if self.aq1_checkpoint:
            from ..models.anemll_quant import load_dynamic_weights_for_ane
            print(f"  Loading AQ1 checkpoint: {self.aq1_checkpoint}")
            load_dynamic_weights_for_ane(model, self.aq1_checkpoint, verbose=True)
            print("  AQ1 weights loaded - Conv2d replaced with AnemllConv2d (custom op)")

        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            layers_per_chunk = total_layers // total_chunks
            start_layer = chunk_idx * layers_per_chunk
            end_layer = min((chunk_idx + 1) * layers_per_chunk, total_layers)
        else:
            start_layer = 0
            end_layer = None

        class FFNWrapper(torch.nn.Module):
            def __init__(self, model: QwenForCausalLM) -> None:
                super().__init__()
                self.model = model  # Use QwenForCausalLM as root
                self.states = QwenConverter.GetTransformerStates(
                    model, part="2", prefix="model.model."
                )

            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                rotary = self.model.model.get_rotary_embeddings_s(current_pos)
                out = self.model.model.process_layers(
                    hidden_states,
                    position_ids,
                    causal_mask,
                    current_pos,
                    rotary,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    IN_PREFILL=False,
                )
                out = self.model.model.norm(out)
                return out

        wrapper = FFNWrapper(model)
        wrapper.eval()

        hidden_states = torch.zeros(
            (1, 1, model.config.hidden_size), dtype=torch.float16, device=TEST_DEVICE
        )
        position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        causal_mask = torch.zeros(
            (1, 1, 1, self.context_length), dtype=torch.float16, device=TEST_DEVICE
        )
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        traced = torch.jit.trace(
            wrapper, (hidden_states, position_ids, causal_mask, current_pos)
        )

        # Dynamic mode: Apply A1F approach to preserve factored scales (A @ B)
        # Use value_inference patch to prevent large tensor folding during frontend
        # Keep const_elimination to properly handle scalar constants (like layer_norm epsilon)
        if use_dynamic:
            # Patch value_inference BEFORE conversion to prevent large const*const folding
            original_value_inference = _patch_value_inference(threshold=100000)

            try:
                mlmodel = ct.convert(
                    traced,
                    inputs=[
                        ct.TensorType(
                            name="hidden_states", shape=hidden_states.shape, dtype=np.float16
                        ),
                        ct.TensorType(
                            name="position_ids", shape=position_ids.shape, dtype=np.int32
                        ),
                        ct.TensorType(
                            name="causal_mask", shape=causal_mask.shape, dtype=np.float16
                        ),
                        ct.TensorType(
                            name="current_pos", shape=current_pos.shape, dtype=np.int32
                        ),
                    ],
                    outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
                    states=self.GetTransformerStates(model, part=None, prefix="model.model."),
                    convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                    minimum_deployment_target=ct.target.iOS18,
                    # Note: Keep DEFAULT pipeline - const_elimination needed for scalar constants (epsilon)
                    # The _patch_value_inference handles selective blocking of large tensor folding
                )
            finally:
                _restore_value_inference(original_value_inference)
        else:
            mlmodel = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(
                        name="hidden_states", shape=hidden_states.shape, dtype=np.float16
                    ),
                    ct.TensorType(
                        name="position_ids", shape=position_ids.shape, dtype=np.int32
                    ),
                    ct.TensorType(
                        name="causal_mask", shape=causal_mask.shape, dtype=np.float16
                    ),
                    ct.TensorType(
                        name="current_pos", shape=current_pos.shape, dtype=np.int32
                    ),
                ],
                outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
                states=self.GetTransformerStates(model, part=None, prefix="model.model."),
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.iOS18,
                convert_to="mlprogram",
            )

        if self.lut_bits and not use_dynamic:
            self.converted_model = mlmodel
            # WORKAROUND: CoreMLTools has a known bug where LUT quantization fails with multiple workers
            # when processing chunked models. The second chunk quantization fails with "Pool not running".
            # Setting workers to None (single-threaded) avoids this issue.
            # TODO: File bug report with Apple CoreMLTools team about multi-worker quantization failure on chunked models
            num_workers = None if total_chunks > 1 else 8
            self.postprocess(num_workers=num_workers)
            mlmodel = self.converted_model

        return mlmodel

    def convert_part_2_prefill(
        self, model: QwenForCausalLM, chunk_idx: int = 0, total_chunks: int = 1
    ) -> ct.models.MLModel:
        """Convert transformer layers for prefill mode."""
        require_coreml()

        # Check if dynamic mode is enabled
        use_dynamic = os.environ.get('ANEMLL_DYNAMIC_SCALES', '0') == '1'
        if use_dynamic:
            from ..models.anemll_quant import ensure_coreml_converter_registered
            ensure_coreml_converter_registered()
            print("  Dynamic mode (prefill): registered CoreML converter for constexpr_lut_to_dense + matmul")

        # Load AQ1 checkpoint if provided - replaces Conv2d with AnemllConv2d
        if self.aq1_checkpoint:
            from ..models.anemll_quant import load_dynamic_weights_for_ane
            print(f"  Loading AQ1 checkpoint (prefill): {self.aq1_checkpoint}")
            load_dynamic_weights_for_ane(model, self.aq1_checkpoint, verbose=True)
            print("  AQ1 weights loaded - Conv2d replaced with AnemllConv2d (custom op)")

        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            layers_per_chunk = total_layers // total_chunks
            start_layer = chunk_idx * layers_per_chunk
            end_layer = min((chunk_idx + 1) * layers_per_chunk, total_layers)
        else:
            start_layer = 0
            end_layer = None

        class PrefillWrapper(torch.nn.Module):
            def __init__(self, model: QwenForCausalLM, start_layer=0, end_layer=None):
                super().__init__()
                self.model = model  # Use QwenForCausalLM as root
                self.start_layer = start_layer
                self.end_layer = end_layer
                self.states = QwenConverter.GetTransformerStates(
                    model, part="2_prefill", prefix="model.model."
                )

            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                rotary = self.model.model.get_rotary_embedding_prefill(position_ids)
                out = self.model.model.process_layers(
                    hidden_states,
                    position_ids,
                    causal_mask,
                    current_pos,
                    rotary,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    IN_PREFILL=True,
                )

                # Skip normalization for prefill - data not used, only KV cache is updated!
                # This follows the LLAMA pattern and avoids unnecessary computation
                if self.end_layer is None or self.end_layer == len(self.model.model.layers):
                    print("Skipping final normalization for prefill, data not used!")
                    # Return only first token to minimize memory usage
                    return out[:, 0:1, :]
                
                return out

        wrapper = PrefillWrapper(model, start_layer, end_layer)
        wrapper.eval()

        # Check if this is the last chunk in a multi-chunk model
        is_last_chunk = (chunk_idx == total_chunks - 1)
        
        hidden_states = torch.zeros(
            (1, self.batch_size, model.config.hidden_size),
            dtype=torch.float16,
            device=TEST_DEVICE,
        )
        position_ids = torch.zeros(
            (self.batch_size,), dtype=torch.int32, device=TEST_DEVICE
        )
        causal_mask = torch.zeros(
            (1, 1, self.batch_size, self.context_length),
            dtype=torch.float16,
            device=TEST_DEVICE,
        )
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        traced = torch.jit.trace(
            wrapper, (hidden_states, position_ids, causal_mask, current_pos)
        )

        # Dynamic mode: Apply A1F approach to preserve factored scales (A @ B)
        # Use value_inference patch to prevent large tensor folding during frontend
        # Keep const_elimination to properly handle scalar constants (like layer_norm epsilon)
        if use_dynamic:
            # Patch value_inference BEFORE conversion to prevent large const*const folding
            original_value_inference = _patch_value_inference(threshold=100000)

            try:
                mlmodel = ct.convert(
                    traced,
                    inputs=[
                        ct.TensorType(
                            name="hidden_states", shape=hidden_states.shape, dtype=np.float16
                        ),
                        ct.TensorType(
                            name="position_ids", shape=position_ids.shape, dtype=np.int32
                        ),
                        ct.TensorType(
                            name="causal_mask", shape=causal_mask.shape, dtype=np.float16
                        ),
                        ct.TensorType(
                            name="current_pos", shape=current_pos.shape, dtype=np.int32
                        ),
                    ],
                    outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
                    states=wrapper.states,
                    convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                    minimum_deployment_target=ct.target.iOS18,
                    # Note: Keep DEFAULT pipeline - const_elimination needed for scalar constants
                )
            finally:
                _restore_value_inference(original_value_inference)
        else:
            mlmodel = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(
                        name="hidden_states", shape=hidden_states.shape, dtype=np.float16
                    ),
                    ct.TensorType(
                        name="position_ids", shape=position_ids.shape, dtype=np.int32
                    ),
                    ct.TensorType(
                        name="causal_mask", shape=causal_mask.shape, dtype=np.float16
                    ),
                    ct.TensorType(
                        name="current_pos", shape=current_pos.shape, dtype=np.int32
                    ),
                ],
                outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
                states=wrapper.states,
                convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.iOS18,
            )

        if self.lut_bits and not use_dynamic:
            self.converted_model = mlmodel
            # WORKAROUND: CoreMLTools has a known bug where LUT quantization fails with multiple workers
            # when processing chunked models. The second chunk quantization fails with "Pool not running".
            # Setting workers to None (single-threaded) avoids this issue.
            # TODO: File bug report with Apple CoreMLTools team about multi-worker quantization failure on chunked models
            num_workers = None if total_chunks > 1 else 8
            self.postprocess(num_workers=num_workers)
            mlmodel = self.converted_model

        return mlmodel

    def convert_prefill(self, model: QwenForCausalLM) -> ct.models.MLModel:
        """Convert Qwen model to CoreML format for prefill mode.

        Args:
            model: The Qwen model to convert

        Returns:
            ct.models.MLModel: Converted model for prefill processing
        """
        require_coreml()
        print("Converting Qwen model for prefill mode...")

        class PrefillWrapper(torch.nn.Module):
            def __init__(
                self, model: QwenForCausalLM, context_length: int, batch_size: int
            ) -> None:
                super().__init__()
                self.model = model
                self.context_length = context_length
                self.batch_size = batch_size

            def forward(
                self,
                hidden_states: torch.Tensor,
                position_ids: torch.Tensor,
                causal_mask: torch.Tensor,
                current_pos: torch.Tensor,
            ) -> torch.Tensor:
                # Prefill mode: only process transformer layers, skip embeddings and LM head
                # This updates KV cache state without generating logits
                return self.model.forward_prefill(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                )

        wrapper = PrefillWrapper(model, self.context_length, self.batch_size)
        wrapper.eval()
        print("Prefill wrapper model created and set to eval mode")

        print("Preparing prefill model inputs for tracing...")
        # Use batch_size for prefill mode (multiple tokens at once)
        # Input is hidden_states instead of input_ids (skip embeddings)
        sample_hidden_states = torch.zeros(
            (1, self.batch_size, model.config.hidden_size),
            dtype=torch.float16,
            device=TEST_DEVICE,
        )  # [1, batch_size, hidden_size]
        sample_position_ids = torch.zeros(
            (self.batch_size,), dtype=torch.int32, device=TEST_DEVICE
        )  # [batch_size]
        sample_causal_mask = torch.zeros(
            (1, 1, self.batch_size, self.context_length),
            dtype=torch.float16,
            device=TEST_DEVICE,
        )  # [1, 1, batch_size, context_length]
        sample_current_pos = torch.zeros(
            (1,), dtype=torch.int32, device=TEST_DEVICE
        )  # [1] - current position

        print("Prefill sample inputs created")
        print(f"sample_hidden_states shape: {sample_hidden_states.shape}")
        print(f"sample_position_ids shape: {sample_position_ids.shape}")
        print(f"sample_causal_mask shape: {sample_causal_mask.shape}")
        print(f"sample_current_pos shape: {sample_current_pos.shape}")

        print("Starting torch.jit.trace for prefill...")
        traced = torch.jit.trace(
            wrapper,
            (
                sample_hidden_states,
                sample_position_ids,
                sample_causal_mask,
                sample_current_pos,
            ),
        )
        print("torch.jit.trace for prefill completed!")

        print("Starting CoreML conversion for prefill...")
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=sample_hidden_states.shape,
                    dtype=np.float16,
                ),
                ct.TensorType(
                    name="position_ids", shape=sample_position_ids.shape, dtype=np.int32
                ),
                ct.TensorType(
                    name="causal_mask", shape=sample_causal_mask.shape, dtype=np.float16
                ),
                ct.TensorType(
                    name="current_pos", shape=sample_current_pos.shape, dtype=np.int32
                ),
            ],
            outputs=[
                ct.TensorType(
                    name="output_hidden_states", dtype=np.float16
                ),  # Only output hidden states, no logits
            ],
            states=self.GetTransformerStates(model, part=None, prefix="model.model."),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        print("CoreML conversion for prefill completed!")

        # Apply LUT quantization if specified
        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)  # Allow passing num_workers if needed
            mlmodel = self.converted_model

        return mlmodel

    def convert_embeddings(self, model: QwenForCausalLM) -> ct.models.MLModel:
        """Convert embeddings layer to CoreML format.

        Args:
            model: The Qwen model containing embeddings

        Returns:
            ct.models.MLModel: Converted CoreML model for embeddings
        """
        require_coreml()
        print("\nConverting Qwen embeddings layer...")

        class EmbeddingsWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.embed_tokens = model.model.embed_tokens

            def forward(self, input_ids):
                hidden_states = self.embed_tokens(input_ids)
                return hidden_states.to(MODEL_DTYPE)

        # Create wrapper and ensure eval mode
        wrapper = EmbeddingsWrapper(model)
        wrapper.eval()

        # Create sample input for tracing
        sample_input = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)

        # Trace model
        print("Tracing embeddings model...")
        traced_model = torch.jit.trace(wrapper, sample_input)

        # Define flexible input shapes for both single token and batch processing
        input_shape = ct.EnumeratedShapes(
            shapes=[
                [1, 1],
                [1, self.batch_size],
            ],  # Support single token and batch_size tokens
            default=[1, 1],  # Use single token as default
        )

        print(f"Converting embeddings model with input shape: {input_shape}")

        # Convert to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=input_shape,  # Use enumerated shapes for flexibility
                    dtype=np.int32,
                )
            ],
            outputs=[ct.TensorType(name="hidden_states", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        print("Embeddings conversion completed")

        # Apply LUT quantization if specified
        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)  # Allow passing num_workers if needed
            mlmodel = self.converted_model

        return mlmodel

    def _load_aq1_data(self):
        """Load AQ1 checkpoint data if not already loaded."""
        if self.aq1_data is None and self.aq1_checkpoint:
            print(f"Loading AQ1 checkpoint: {self.aq1_checkpoint}")
            self.aq1_data = load_aq1_checkpoint(
                self.aq1_checkpoint,
                nbits_mlp=self.aq1_nbits_mlp,
                nbits_attn=self.aq1_nbits_attn,
                max_layers=self.num_layers,  # Filter layers during loading
                verbose=True,
            )
        return self.aq1_data

    def convert_part_2_aq1_mlp(
        self, model: QwenForCausalLM, chunk_idx: int = 0, total_chunks: int = 1
    ) -> ct.models.MLModel:
        """Convert MLP layers using AQ1 MIL builder with constexpr_lut_to_dense.

        This method builds the MIL program directly using constexpr_lut_to_dense
        for base weights and matmul for dynamic A*B scale computation.

        Unlike PyTorch tracing, this preserves the quantization structure:
        - W_base = constexpr_lut_to_dense(packed_indices, LUT)
        - scales = matmul(A, B)
        - W_eff = W_base * scales
        - output = conv(x, W_eff)

        Args:
            model: The Qwen model (used for config)
            chunk_idx: Chunk index for multi-chunk conversion
            total_chunks: Total number of chunks

        Returns:
            ct.models.MLModel: Converted model with dynamic AQ1 quantization
        """
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types

        require_coreml()

        # Load AQ1 data
        aq1_data = self._load_aq1_data()
        if not aq1_data:
            raise ValueError("AQ1 checkpoint required for AQ1 conversion")

        # Determine layer range for this chunk
        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            layers_per_chunk = total_layers // total_chunks
            start_layer = chunk_idx * layers_per_chunk
            end_layer = min((chunk_idx + 1) * layers_per_chunk, total_layers)
        else:
            start_layer = 0
            end_layer = total_layers

        print(f"Converting AQ1 MLP layers {start_layer} to {end_layer} "
              f"(chunk {chunk_idx + 1}/{total_chunks})")

        hidden_size = model.config.hidden_size
        intermediate_size = model.config.intermediate_size

        # Build MIL program for MLP layers
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, hidden_size, 1, 1), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18  # iOS18 required for constexpr_lut_to_dense with sub-byte types
        )
        def prog(hidden_states):
            x = hidden_states

            for layer_idx in range(start_layer, end_layer):
                base_name = f"model.layers.{layer_idx}.mlp"

                # Get AQ1 data for each projection
                gate_data = get_layer_aq1_data(aq1_data, f"{base_name}.gate_proj")
                up_data = get_layer_aq1_data(aq1_data, f"{base_name}.up_proj")
                down_data = get_layer_aq1_data(aq1_data, f"{base_name}.down_proj")

                if gate_data is None or up_data is None or down_data is None:
                    raise ValueError(f"Missing AQ1 data for MLP layer {layer_idx}")

                # Gate projection with AQ1
                gate = build_aq1_conv_layer(
                    x=x,
                    indices_packed=gate_data['indices_packed'],
                    lut=gate_data['lut'],
                    scale_A=gate_data['scale_A'],
                    scale_B=gate_data['scale_B'],
                    out_features=gate_data['out_features'],
                    in_features=gate_data['in_features'],
                    name=f"layer{layer_idx}_gate_proj"
                )

                # Up projection with AQ1
                up = build_aq1_conv_layer(
                    x=x,
                    indices_packed=up_data['indices_packed'],
                    lut=up_data['lut'],
                    scale_A=up_data['scale_A'],
                    scale_B=up_data['scale_B'],
                    out_features=up_data['out_features'],
                    in_features=up_data['in_features'],
                    name=f"layer{layer_idx}_up_proj"
                )

                # SiLU activation on gate
                gate_silu = mb.silu(x=gate, name=f"layer{layer_idx}_gate_silu")

                # Multiply gate * up
                hidden = mb.mul(x=gate_silu, y=up, name=f"layer{layer_idx}_hidden")

                # Down projection with AQ1
                mlp_out = build_aq1_conv_layer(
                    x=hidden,
                    indices_packed=down_data['indices_packed'],
                    lut=down_data['lut'],
                    scale_A=down_data['scale_A'],
                    scale_B=down_data['scale_B'],
                    out_features=down_data['out_features'],
                    in_features=down_data['in_features'],
                    name=f"layer{layer_idx}_down_proj"
                )

                # Residual connection
                x = mb.add(x=x, y=mlp_out, name=f"layer{layer_idx}_residual")

            return x

        # Convert with EMPTY pipeline to preserve dynamic ops
        print("Converting MIL program to CoreML (with pass_pipeline=EMPTY)...")
        mlmodel = ct.convert(
            prog,
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,  # iOS18 for sub-byte constexpr_lut_to_dense
            pass_pipeline=ct.PassPipeline.EMPTY,  # Critical: prevent constant folding
        )

        # Verify the model structure
        print("\nVerifying AQ1 model structure...")
        spec = mlmodel.get_spec()
        if hasattr(spec, 'mlProgram'):
            prog_spec = spec.mlProgram
            constexpr_count = 0
            matmul_count = 0
            for func_name, func in prog_spec.functions.items():
                for block_name, block in func.block_specializations.items():
                    for op in block.operations:
                        if op.type == 'constexpr_lut_to_dense':
                            constexpr_count += 1
                        elif op.type == 'matmul':
                            matmul_count += 1
            print(f"  constexpr_lut_to_dense ops: {constexpr_count}")
            print(f"  matmul ops (includes A@B): {matmul_count}")

            # Expected: 3 constexpr per layer (gate, up, down), 3 matmul per layer (A@B)
            expected_constexpr = (end_layer - start_layer) * 3
            expected_matmul = (end_layer - start_layer) * 3
            if constexpr_count == expected_constexpr and matmul_count >= expected_matmul:
                print("  ✓ AQ1 structure verified: dynamic A*B preserved!")
            else:
                print(f"  ⚠ Expected {expected_constexpr} constexpr, {expected_matmul} matmul")

        return mlmodel

    def convert_part_2_aq1_conv(
        self, model: QwenForCausalLM, chunk_idx: int = 0, total_chunks: int = 1
    ) -> ct.models.MLModel:
        """Convert MLP layers using AnemllConv2d.build_mil_conv().

        This method creates AnemllConv2d layers from checkpoint data and uses
        their build_mil_conv() method to generate the MIL program.

        Benefits over convert_part_2_aq1_mlp:
        - Cleaner API: layers store their own indices, lut, scale_A, scale_B
        - Reusable: AnemllConv2d can be used for training and inference
        - Consistent: Same layer class works in PyTorch and CoreML

        Args:
            model: The Qwen model (used for config)
            chunk_idx: Chunk index for multi-chunk conversion
            total_chunks: Total number of chunks

        Returns:
            ct.models.MLModel: Converted model with dynamic AQ1 quantization
        """
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types

        require_coreml()

        # Load AQ1 data
        aq1_data = self._load_aq1_data()
        if not aq1_data:
            raise ValueError("AQ1 checkpoint required for AQ1 conversion. Use --aq1-checkpoint")

        # Determine layer range for this chunk
        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            layers_per_chunk = total_layers // total_chunks
            start_layer = chunk_idx * layers_per_chunk
            end_layer = min((chunk_idx + 1) * layers_per_chunk, total_layers)
        else:
            start_layer = 0
            end_layer = total_layers

        print(f"Converting AQ1 MLP layers {start_layer} to {end_layer} "
              f"(chunk {chunk_idx + 1}/{total_chunks}) using AnemllConv2d")

        hidden_size = model.config.hidden_size
        intermediate_size = model.config.intermediate_size

        # Create AnemllConv2d layers from checkpoint data
        mlp_layers = {}
        for layer_idx in range(start_layer, end_layer):
            base_name = f"model.layers.{layer_idx}.mlp"

            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                full_name = f"{base_name}.{proj_name}"
                data = get_layer_aq1_data(aq1_data, full_name)

                if data is None:
                    raise ValueError(f"Missing AQ1 data for {full_name}")

                # Create AnemllConv2d from checkpoint data
                conv_layer = AnemllConv2d(
                    in_features=data['in_features'],
                    out_features=data['out_features'],
                    scale_rank=data['scale_A'].shape[1],
                    bias=False,
                    dtype=torch.float16,
                    lut_bits=data['nbits'],
                )

                # Set LUT
                conv_layer.set_lut(torch.from_numpy(data['lut']))

                # Set scale factors
                conv_layer.scale_A.data.copy_(torch.from_numpy(data['scale_A']))
                conv_layer.scale_B.data.copy_(torch.from_numpy(data['scale_B']))

                # Set indices from packed data (unpack first)
                # Actually, we need to store packed indices directly for build_mil_conv
                # The build_mil_conv uses get_packed_indices() which will repack
                # For efficiency, we can set _indices_packed directly
                conv_layer._indices_packed = data['indices_packed']

                # Also need to set indices for potential PyTorch use
                # Unpack to set indices (for verification)
                nbits = data['nbits']
                indices_per_byte = 8 // nbits
                packed = data['indices_packed']
                total_elements = data['out_features'] * data['in_features']

                unpacked = []
                mask = (1 << nbits) - 1
                for byte in packed:
                    for i in range(indices_per_byte):
                        if len(unpacked) < total_elements:
                            unpacked.append((byte >> (i * nbits)) & mask)

                indices_tensor = torch.tensor(unpacked, dtype=torch.long).view(
                    data['out_features'], data['in_features']
                )
                conv_layer.indices.copy_(indices_tensor)

                mlp_layers[full_name] = conv_layer

        print(f"  Created {len(mlp_layers)} AnemllConv2d layers")

        # Build MIL program using AnemllConv2d.build_mil_conv()
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, hidden_size, 1, 1), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18  # iOS18 required for constexpr_lut_to_dense with sub-byte types
        )
        def prog(hidden_states):
            x = hidden_states

            for layer_idx in range(start_layer, end_layer):
                base_name = f"model.layers.{layer_idx}.mlp"

                gate_layer = mlp_layers[f"{base_name}.gate_proj"]
                up_layer = mlp_layers[f"{base_name}.up_proj"]
                down_layer = mlp_layers[f"{base_name}.down_proj"]

                # Gate projection using build_mil_conv
                gate = gate_layer.build_mil_conv(x, name=f"layer{layer_idx}_gate_proj")

                # Up projection using build_mil_conv
                up = up_layer.build_mil_conv(x, name=f"layer{layer_idx}_up_proj")

                # SiLU activation on gate
                gate_silu = mb.silu(x=gate, name=f"layer{layer_idx}_gate_silu")

                # Multiply gate * up
                hidden = mb.mul(x=gate_silu, y=up, name=f"layer{layer_idx}_hidden")

                # Down projection using build_mil_conv
                mlp_out = down_layer.build_mil_conv(hidden, name=f"layer{layer_idx}_down_proj")

                # Residual connection
                x = mb.add(x=x, y=mlp_out, name=f"layer{layer_idx}_residual")

            return x

        # Convert with EMPTY pipeline to preserve dynamic ops
        print("Converting MIL program to CoreML (with pass_pipeline=EMPTY)...")
        mlmodel = ct.convert(
            prog,
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,  # iOS18 for sub-byte constexpr_lut_to_dense
            pass_pipeline=ct.PassPipeline.EMPTY,
        )

        # Verify the model structure
        print("\nVerifying AQ1 model structure (AnemllConv2d)...")
        spec = mlmodel.get_spec()
        if hasattr(spec, 'mlProgram'):
            prog_spec = spec.mlProgram
            constexpr_count = 0
            matmul_count = 0
            for func_name, func in prog_spec.functions.items():
                for block_name, block in func.block_specializations.items():
                    for op in block.operations:
                        if op.type == 'constexpr_lut_to_dense':
                            constexpr_count += 1
                        elif op.type == 'matmul':
                            matmul_count += 1
            print(f"  constexpr_lut_to_dense ops: {constexpr_count}")
            print(f"  matmul ops (includes A@B): {matmul_count}")

            expected_constexpr = (end_layer - start_layer) * 3
            expected_matmul = (end_layer - start_layer) * 3
            if constexpr_count == expected_constexpr and matmul_count >= expected_matmul:
                print("  ✓ AQ1 structure verified: dynamic A*B preserved!")
            else:
                print(f"  ⚠ Expected {expected_constexpr} constexpr, {expected_matmul} matmul")

        return mlmodel

    # =========================================================================
    # AQ1 Full Transformer: MIL-based with attention + KV cache + MLP
    # =========================================================================

    def convert_part_2_aq1_full(
        self, model: QwenForCausalLM, chunk_idx: int = 0, total_chunks: int = 1
    ) -> ct.models.MLModel:
        """Convert full transformer layers with AQ1 quantization using direct MIL.

        This method builds a complete transformer in MIL with:
        - constexpr_lut_to_dense for LUT-indexed weights
        - matmul for dynamic A@B scale computation
        - Full attention with Q/K/V/O projections
        - KV cache states for generation
        - MLP with gate/up/down projections
        - pass_pipeline=EMPTY to prevent constant folding

        Args:
            model: The Qwen model (used for config)
            chunk_idx: Chunk index for multi-chunk conversion
            total_chunks: Total number of chunks

        Returns:
            ct.models.MLModel: Converted model with AQ1 quantization preserved
        """
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        from ..models.aq1_mil_layer import build_aq1_conv_layer, load_aq1_checkpoint, get_layer_aq1_data

        require_coreml()

        # Load AQ1 data
        aq1_data = self._load_aq1_data()
        if not aq1_data:
            raise ValueError("AQ1 checkpoint required. Use --aq1-checkpoint")

        # Get model config
        config = model.config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", hidden_size // num_heads)
        rms_norm_eps = config.rms_norm_eps

        # Determine layer range for this chunk
        total_layers = config.num_hidden_layers
        if self.num_layers is not None:
            total_layers = min(self.num_layers, total_layers)

        if total_chunks > 1:
            layers_per_chunk = total_layers // total_chunks
            start_layer = chunk_idx * layers_per_chunk
            end_layer = min((chunk_idx + 1) * layers_per_chunk, total_layers)
        else:
            start_layer = 0
            end_layer = total_layers

        num_layers = end_layer - start_layer
        context_length = self.context_length

        print(f"\n[AQ1 Full] Converting layers {start_layer} to {end_layer-1}")
        print(f"  hidden_size={hidden_size}, intermediate={intermediate_size}")
        print(f"  num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        print(f"  context_length={context_length}")

        # Load layer norms from the original model
        layer_norms = {}
        for layer_idx in range(start_layer, end_layer):
            prefix = f"model.layers.{layer_idx}"
            layer_norms[layer_idx] = {
                'input_layernorm': model.model.layers[layer_idx].input_layernorm.weight.detach().cpu().numpy().astype(np.float16),
                'post_attention_layernorm': model.model.layers[layer_idx].post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float16),
            }
            # Load q_norm and k_norm if they exist
            if hasattr(model.model.layers[layer_idx].self_attn, 'q_norm'):
                layer_norms[layer_idx]['q_norm'] = model.model.layers[layer_idx].self_attn.q_norm.weight.detach().cpu().numpy().astype(np.float16)
                layer_norms[layer_idx]['k_norm'] = model.model.layers[layer_idx].self_attn.k_norm.weight.detach().cpu().numpy().astype(np.float16)

        # Load final norm
        final_norm_weight = model.model.norm.weight.detach().cpu().numpy().astype(np.float16)

        # Load RoPE frequencies from model
        # For Qwen, we need to compute RoPE cos/sin tables
        rope_theta = getattr(config, 'rope_theta', 10000.0)

        # Build RoPE embedding table - use 2x context_length to match working model
        max_rope_len = context_length * 2
        inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        positions = np.arange(max_rope_len, dtype=np.float32)
        freqs = np.outer(positions, inv_freq)  # [max_rope_len, head_dim/2]
        cos_cached = np.cos(freqs).astype(np.float16)  # [max_rope_len, head_dim/2]
        sin_cached = np.sin(freqs).astype(np.float16)  # [max_rope_len, head_dim/2]

        # Expand to full head_dim (interleaved) with batch dimension [1, max_rope_len, head_dim]
        cos_full = np.zeros((1, max_rope_len, head_dim), dtype=np.float16)
        sin_full = np.zeros((1, max_rope_len, head_dim), dtype=np.float16)
        cos_full[0, :, 0::2] = cos_cached
        cos_full[0, :, 1::2] = cos_cached
        sin_full[0, :, 0::2] = sin_cached
        sin_full[0, :, 1::2] = sin_cached

        print(f"  RoPE table shape: cos={cos_full.shape}, sin={sin_full.shape}")

        # Create AnemllConv2d layers for all projections
        all_layers = {}
        for layer_idx in range(start_layer, end_layer):
            base = f"model.layers.{layer_idx}"
            all_layers[layer_idx] = {}

            # MLP projections
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                name = f"{base}.mlp.{proj}"
                data = get_layer_aq1_data(aq1_data, name)
                if data is None:
                    raise ValueError(f"Missing AQ1 data for {name}")
                all_layers[layer_idx][f'mlp.{proj}'] = data

            # Attention projections
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                name = f"{base}.self_attn.{proj}"
                data = get_layer_aq1_data(aq1_data, name)
                if data is None:
                    raise ValueError(f"Missing AQ1 data for {name}")
                all_layers[layer_idx][f'attn.{proj}'] = data

        print(f"  Loaded AQ1 data for {len(all_layers)} layers")

        # Helper to build RMSNorm in MIL
        def build_rmsnorm(x, weight, eps, name):
            """Build RMSNorm: x * rsqrt(mean(x^2) + eps) * weight"""
            # Subtract mean first (ANE compatibility)
            mean = mb.reduce_mean(x=x, axes=[-1], keep_dims=True, name=f"{name}_mean")
            x_centered = mb.sub(x=x, y=mean, name=f"{name}_centered")

            # Compute variance
            x_sq = mb.mul(x=x_centered, y=x_centered, name=f"{name}_sq")
            var = mb.reduce_mean(x=x_sq, axes=[-1], keep_dims=True, name=f"{name}_var")

            # Add eps and rsqrt
            var_eps = mb.add(x=var, y=np.float16(eps), name=f"{name}_var_eps")
            rsqrt = mb.rsqrt(x=var_eps, name=f"{name}_rsqrt")

            # Normalize and scale
            normalized = mb.mul(x=x_centered, y=rsqrt, name=f"{name}_norm")
            weight_const = mb.const(val=weight, name=f"{name}_weight")
            return mb.mul(x=normalized, y=weight_const, name=f"{name}_out")

        # Build the MIL program with combined KV cache state
        # Combined KV cache: [num_layers * 2, num_kv_heads, context_length, head_dim]
        # Layout: [k0, v0, k1, v1, ...] where k0/v0 is layer 0's K/V cache
        kv_cache_shape = (num_layers * 2, num_kv_heads, context_length, head_dim)

        print(f"  Combined KV cache shape: {kv_cache_shape}")
        print(f"  Layout: [K0, V0, K1, V1, ...] for {num_layers} layers")

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, hidden_size, 1, 1), dtype=types.fp16),  # hidden_states
                mb.TensorSpec(shape=(1,), dtype=types.int32),  # position_ids
                mb.TensorSpec(shape=(1, 1, 1, context_length), dtype=types.fp16),  # causal_mask
                mb.StateTensorSpec(shape=kv_cache_shape, dtype=types.fp16),  # kv_cache
            ],
            opset_version=ct.target.iOS18
        )
        def prog(hidden_states, position_ids, causal_mask, kv_cache_state):
            x = hidden_states  # [1, hidden_size, 1, 1]

            # Read current KV cache
            kv_cache = mb.read_state(input=kv_cache_state, name="kv_cache_read")
            # kv_cache: [num_layers * 2, num_kv_heads, context_length, head_dim]

            # Reshape to [1, 1, hidden_size] for transformer ops
            x = mb.reshape(x=x, shape=[1, 1, hidden_size], name="input_reshape")

            # Collect updated KV slices
            # kv_cache_running will be updated via chained slice_updates (no slice_by_index, no concat)
            kv_cache_running = kv_cache

            # Position preprocessing chain (matches traced model pattern for ANE compatibility)
            # This greater_equal -> add -> select chain helps ANE dispatch the subsequent cast
            # Done ONCE before layer loop, like the traced model
            pos_ge_zero = mb.greater_equal(x=position_ids, y=np.int32(0), name="pos_ge_zero")
            pos_plus_2048 = mb.add(x=position_ids, y=np.int32(context_length * 2), name="pos_plus_2048")
            pos_selected = mb.select(a=position_ids, b=pos_plus_2048, cond=pos_ge_zero, name="pos_selected")

            # Cast to uint16 for gather (ANE requires uint16 indices)
            pos_uint16 = mb.cast(x=pos_selected, dtype="uint16", name="pos_uint16")

            for layer_offset in range(num_layers):
                layer_idx = start_layer + layer_offset
                layer_data = all_layers[layer_idx]
                norms = layer_norms[layer_idx]

                # Store residual
                residual = x

                # ========== Input LayerNorm ==========
                x = build_rmsnorm(x, norms['input_layernorm'], rms_norm_eps, f"layer{layer_idx}_input_norm")

                # Reshape for conv: [1, 1, hidden_size] -> [1, hidden_size, 1, 1]
                x_conv = mb.reshape(x=x, shape=[1, hidden_size, 1, 1], name=f"layer{layer_idx}_attn_in_reshape")

                # ========== Q/K/V Projections ==========
                q_data = layer_data['attn.q_proj']
                k_data = layer_data['attn.k_proj']
                v_data = layer_data['attn.v_proj']

                q = build_aq1_conv_layer(
                    x_conv, q_data['indices_packed'], q_data['lut'],
                    q_data['scale_A'], q_data['scale_B'],
                    q_data['out_features'], q_data['in_features'],
                    name=f"layer{layer_idx}_q_proj"
                )
                k = build_aq1_conv_layer(
                    x_conv, k_data['indices_packed'], k_data['lut'],
                    k_data['scale_A'], k_data['scale_B'],
                    k_data['out_features'], k_data['in_features'],
                    name=f"layer{layer_idx}_k_proj"
                )
                v = build_aq1_conv_layer(
                    x_conv, v_data['indices_packed'], v_data['lut'],
                    v_data['scale_A'], v_data['scale_B'],
                    v_data['out_features'], v_data['in_features'],
                    name=f"layer{layer_idx}_v_proj"
                )

                # Reshape Q/K/V: [1, heads*head_dim, 1, 1] -> [1, heads, 1, head_dim]
                q = mb.reshape(x=q, shape=[1, num_heads, 1, head_dim], name=f"layer{layer_idx}_q_reshape")
                k = mb.reshape(x=k, shape=[1, num_kv_heads, 1, head_dim], name=f"layer{layer_idx}_k_reshape")
                v = mb.reshape(x=v, shape=[1, num_kv_heads, 1, head_dim], name=f"layer{layer_idx}_v_reshape")

                # ========== Q/K Head Norms (if present) ==========
                if 'q_norm' in norms:
                    q_norm_w = mb.const(val=norms['q_norm'], name=f"layer{layer_idx}_q_norm_w")
                    k_norm_w = mb.const(val=norms['k_norm'], name=f"layer{layer_idx}_k_norm_w")

                    # Simple scaling (Qwen uses weight-only norm)
                    q = mb.mul(x=q, y=q_norm_w, name=f"layer{layer_idx}_q_normed")
                    k = mb.mul(x=k, y=k_norm_w, name=f"layer{layer_idx}_k_normed")

                # ========== RoPE Embedding ==========
                # Get cos/sin for current position
                # cos_full/sin_full shape: [1, max_rope_len, head_dim]
                cos_table = mb.const(val=cos_full, name=f"layer{layer_idx}_cos_table")
                sin_table = mb.const(val=sin_full, name=f"layer{layer_idx}_sin_table")

                # Gather cos/sin for this position on axis=1 (using shared pos_uint16 from before loop)
                # position_ids is [1], gather gives [1, 1, head_dim]
                cos_pos = mb.gather(x=cos_table, indices=pos_uint16, axis=1, name=f"layer{layer_idx}_cos_pos")
                sin_pos = mb.gather(x=sin_table, indices=pos_uint16, axis=1, name=f"layer{layer_idx}_sin_pos")

                # Reshape for broadcast: [1, 1, head_dim] -> [1, 1, 1, head_dim]
                cos_pos = mb.reshape(x=cos_pos, shape=[1, 1, 1, head_dim], name=f"layer{layer_idx}_cos_reshape")
                sin_pos = mb.reshape(x=sin_pos, shape=[1, 1, 1, head_dim], name=f"layer{layer_idx}_sin_reshape")

                # Apply RoPE: q_rot = q * cos + rotate_half(q) * sin
                # rotate_half: [a, b, c, d] -> [-b, a, -d, c]
                # Split q into even/odd
                q_even = mb.slice_by_index(x=q, begin=[0, 0, 0, 0], end=[1, num_heads, 1, head_dim], stride=[1, 1, 1, 2], name=f"layer{layer_idx}_q_even")
                q_odd = mb.slice_by_index(x=q, begin=[0, 0, 0, 1], end=[1, num_heads, 1, head_dim], stride=[1, 1, 1, 2], name=f"layer{layer_idx}_q_odd")
                q_rotated = mb.concat(values=[mb.mul(x=q_odd, y=np.float16(-1)), q_even], axis=-1, interleave=True, name=f"layer{layer_idx}_q_rotated")

                q_rope = mb.add(
                    x=mb.mul(x=q, y=cos_pos, name=f"layer{layer_idx}_q_cos"),
                    y=mb.mul(x=q_rotated, y=sin_pos, name=f"layer{layer_idx}_q_sin"),
                    name=f"layer{layer_idx}_q_rope"
                )

                # Same for K
                k_even = mb.slice_by_index(x=k, begin=[0, 0, 0, 0], end=[1, num_kv_heads, 1, head_dim], stride=[1, 1, 1, 2], name=f"layer{layer_idx}_k_even")
                k_odd = mb.slice_by_index(x=k, begin=[0, 0, 0, 1], end=[1, num_kv_heads, 1, head_dim], stride=[1, 1, 1, 2], name=f"layer{layer_idx}_k_odd")
                k_rotated = mb.concat(values=[mb.mul(x=k_odd, y=np.float16(-1)), k_even], axis=-1, interleave=True, name=f"layer{layer_idx}_k_rotated")

                k_rope = mb.add(
                    x=mb.mul(x=k, y=cos_pos, name=f"layer{layer_idx}_k_cos"),
                    y=mb.mul(x=k_rotated, y=sin_pos, name=f"layer{layer_idx}_k_sin"),
                    name=f"layer{layer_idx}_k_rope"
                )

                # ========== KV Cache Update (slice_update) ==========
                # ANE-compatible pattern: read_state -> slice_update -> write_state -> read_state
                # Each slice_update MUST output ONLY to write_state, then read again
                # kv_cache_running shape: [num_layers * 2, num_kv_heads, context_length, head_dim]

                # Build begin/end indices for K slice_update
                # IMPORTANT: All values must be int32[1] (1D array), not scalar int32
                zero = mb.const(val=np.array([0], dtype=np.int32), name=f"layer{layer_idx}_zero")
                one_arr = mb.const(val=np.array([1], dtype=np.int32), name=f"layer{layer_idx}_one")
                pos_plus1 = mb.add(x=position_ids, y=one_arr, name=f"layer{layer_idx}_pos_plus1")

                k_layer_idx = mb.const(val=np.array([layer_offset * 2], dtype=np.int32), name=f"layer{layer_idx}_k_layer_idx")
                k_layer_idx_p1 = mb.const(val=np.array([layer_offset * 2 + 1], dtype=np.int32), name=f"layer{layer_idx}_k_layer_p1")
                k_begin = mb.concat(values=[k_layer_idx, zero, position_ids, zero], axis=0, name=f"layer{layer_idx}_k_begin")
                k_end = mb.concat(values=[k_layer_idx_p1, zero, pos_plus1, zero], axis=0, name=f"layer{layer_idx}_k_end")

                # K slice_update -> write_state immediately
                k_cache_updated = mb.slice_update(
                    x=kv_cache_running,
                    update=k_rope,
                    begin=k_begin,
                    end=k_end,
                    stride=[1, 1, 1, 1],
                    begin_mask=[False, False, False, False],
                    end_mask=[False, True, False, True],
                    squeeze_mask=[False, False, False, False],
                    name=f"layer{layer_idx}_k_cache_update"
                )
                # Write K update to state immediately (ANE pattern)
                mb.coreml_update_state(state=kv_cache_state, value=k_cache_updated, name=f"layer{layer_idx}_k_write_state")
                # Read state again for V update
                kv_cache_running = mb.read_state(input=kv_cache_state, name=f"layer{layer_idx}_kv_read_after_k")

                # Build begin/end indices for V slice_update
                v_layer_idx = mb.const(val=np.array([layer_offset * 2 + 1], dtype=np.int32), name=f"layer{layer_idx}_v_layer_idx")
                v_layer_idx_p1 = mb.const(val=np.array([layer_offset * 2 + 2], dtype=np.int32), name=f"layer{layer_idx}_v_layer_p1")
                v_begin = mb.concat(values=[v_layer_idx, zero, position_ids, zero], axis=0, name=f"layer{layer_idx}_v_begin")
                v_end = mb.concat(values=[v_layer_idx_p1, zero, pos_plus1, zero], axis=0, name=f"layer{layer_idx}_v_end")

                # V slice_update -> write_state immediately
                v_cache_updated = mb.slice_update(
                    x=kv_cache_running,
                    update=v,
                    begin=v_begin,
                    end=v_end,
                    stride=[1, 1, 1, 1],
                    begin_mask=[False, False, False, False],
                    end_mask=[False, True, False, True],
                    squeeze_mask=[False, False, False, False],
                    name=f"layer{layer_idx}_v_cache_update"
                )
                # Write V update to state immediately (ANE pattern)
                mb.coreml_update_state(state=kv_cache_state, value=v_cache_updated, name=f"layer{layer_idx}_v_write_state")
                # Read state again for attention computation (and next layer)
                kv_cache_running = mb.read_state(input=kv_cache_state, name=f"layer{layer_idx}_kv_read_after_v")

                # ========== Attention Computation ==========
                # Read K/V for attention from updated cache using STATIC slice_by_index
                k_cache_for_attn = mb.slice_by_index(
                    x=kv_cache_running,
                    begin=[layer_offset * 2, 0, 0, 0],
                    end=[layer_offset * 2 + 1, num_kv_heads, context_length, head_dim],
                    name=f"layer{layer_idx}_k_for_attn"
                )
                v_cache_for_attn = mb.slice_by_index(
                    x=kv_cache_running,
                    begin=[layer_offset * 2 + 1, 0, 0, 0],
                    end=[layer_offset * 2 + 2, num_kv_heads, context_length, head_dim],
                    name=f"layer{layer_idx}_v_for_attn"
                )
                # Reshape to [1, num_kv_heads, context_length, head_dim]
                k_cache_for_attn = mb.reshape(x=k_cache_for_attn, shape=[1, num_kv_heads, context_length, head_dim], name=f"layer{layer_idx}_k_attn_reshape")
                v_cache_for_attn = mb.reshape(x=v_cache_for_attn, shape=[1, num_kv_heads, context_length, head_dim], name=f"layer{layer_idx}_v_attn_reshape")

                # Repeat KV heads for GQA if needed
                n_rep = num_heads // num_kv_heads
                if n_rep > 1:
                    k_for_attn = mb.tile(x=k_cache_for_attn, reps=[1, n_rep, 1, 1], name=f"layer{layer_idx}_k_repeat")
                    v_for_attn = mb.tile(x=v_cache_for_attn, reps=[1, n_rep, 1, 1], name=f"layer{layer_idx}_v_repeat")
                else:
                    k_for_attn = k_cache_for_attn
                    v_for_attn = v_cache_for_attn

                # Attention scores: Q @ K^T
                # q_rope: [1, num_heads, 1, head_dim]
                # k_for_attn: [1, num_heads, context_length, head_dim] -> [1, num_heads, head_dim, context_length]
                k_t = mb.transpose(x=k_for_attn, perm=[0, 1, 3, 2], name=f"layer{layer_idx}_k_transpose")

                scale = np.float16(1.0 / np.sqrt(head_dim))
                attn_weights = mb.matmul(x=q_rope, y=k_t, name=f"layer{layer_idx}_attn_scores")
                attn_weights = mb.mul(x=attn_weights, y=scale, name=f"layer{layer_idx}_attn_scaled")
                # attn_weights: [1, num_heads, 1, context_length]

                # Apply causal mask
                # causal_mask: [1, 1, 1, context_length]
                attn_weights = mb.add(x=attn_weights, y=causal_mask, name=f"layer{layer_idx}_attn_masked")

                # Softmax
                attn_probs = mb.softmax(x=attn_weights, axis=-1, name=f"layer{layer_idx}_attn_probs")

                # Attention output: probs @ V
                # attn_probs: [1, num_heads, 1, context_length]
                # v_for_attn: [1, num_heads, context_length, head_dim]
                attn_out = mb.matmul(x=attn_probs, y=v_for_attn, name=f"layer{layer_idx}_attn_out")

                # Reshape: [1, num_heads, 1, head_dim] -> [1, num_heads*head_dim, 1, 1]
                attn_out = mb.reshape(x=attn_out, shape=[1, num_heads * head_dim, 1, 1], name=f"layer{layer_idx}_attn_reshape")

                # ========== O Projection ==========
                o_data = layer_data['attn.o_proj']
                attn_out = build_aq1_conv_layer(
                    attn_out, o_data['indices_packed'], o_data['lut'],
                    o_data['scale_A'], o_data['scale_B'],
                    o_data['out_features'], o_data['in_features'],
                    name=f"layer{layer_idx}_o_proj"
                )

                # Reshape to [1, 1, hidden_size] and add residual
                attn_out = mb.reshape(x=attn_out, shape=[1, 1, hidden_size], name=f"layer{layer_idx}_attn_out_reshape")
                x = mb.add(x=residual, y=attn_out, name=f"layer{layer_idx}_attn_residual")

                # ========== Post-Attention LayerNorm ==========
                residual = x
                x = build_rmsnorm(x, norms['post_attention_layernorm'], rms_norm_eps, f"layer{layer_idx}_post_attn_norm")

                # ========== MLP ==========
                # Reshape for conv
                x_conv = mb.reshape(x=x, shape=[1, hidden_size, 1, 1], name=f"layer{layer_idx}_mlp_in_reshape")

                gate_data = layer_data['mlp.gate_proj']
                up_data = layer_data['mlp.up_proj']
                down_data = layer_data['mlp.down_proj']

                gate = build_aq1_conv_layer(
                    x_conv, gate_data['indices_packed'], gate_data['lut'],
                    gate_data['scale_A'], gate_data['scale_B'],
                    gate_data['out_features'], gate_data['in_features'],
                    name=f"layer{layer_idx}_gate_proj"
                )
                up = build_aq1_conv_layer(
                    x_conv, up_data['indices_packed'], up_data['lut'],
                    up_data['scale_A'], up_data['scale_B'],
                    up_data['out_features'], up_data['in_features'],
                    name=f"layer{layer_idx}_up_proj"
                )

                # SiLU activation on gate
                gate_silu = mb.silu(x=gate, name=f"layer{layer_idx}_gate_silu")

                # Multiply gate * up
                hidden = mb.mul(x=gate_silu, y=up, name=f"layer{layer_idx}_hidden")

                # Down projection
                down_out = build_aq1_conv_layer(
                    hidden, down_data['indices_packed'], down_data['lut'],
                    down_data['scale_A'], down_data['scale_B'],
                    down_data['out_features'], down_data['in_features'],
                    name=f"layer{layer_idx}_down_proj"
                )

                # Reshape and add residual
                down_out = mb.reshape(x=down_out, shape=[1, 1, hidden_size], name=f"layer{layer_idx}_mlp_out_reshape")
                x = mb.add(x=residual, y=down_out, name=f"layer{layer_idx}_mlp_residual")

            # NOTE: KV cache state already updated after each slice_update (ANE pattern)
            # No final coreml_update_state needed here

            # ========== Final LayerNorm ==========
            x = build_rmsnorm(x, final_norm_weight, rms_norm_eps, "final_norm")

            # Reshape output: [1, 1, hidden_size] -> [1, hidden_size, 1, 1]
            output = mb.reshape(x=x, shape=[1, hidden_size, 1, 1], name="output_reshape")

            return output

        # Convert with EMPTY pipeline to prevent constant folding
        # StateTensorSpec inputs are automatically converted to CoreML states
        print("\nConverting MIL program to CoreML (pass_pipeline=EMPTY)...")
        print(f"  Using combined StateTensorSpec for KV cache")
        mlmodel = ct.convert(
            prog,
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,
            pass_pipeline=ct.PassPipeline.EMPTY,
        )

        # Verify structure
        print("\n[AQ1 Full] Verifying model structure...")
        spec = mlmodel.get_spec()
        if hasattr(spec, 'mlProgram'):
            prog_spec = spec.mlProgram
            constexpr_count = 0
            matmul_count = 0
            conv_count = 0
            read_state_count = 0
            update_state_count = 0
            scatter_count = 0
            for func_name, func in prog_spec.functions.items():
                for block_name, block in func.block_specializations.items():
                    for op in block.operations:
                        if op.type == 'constexpr_lut_to_dense':
                            constexpr_count += 1
                        elif op.type == 'matmul':
                            matmul_count += 1
                        elif op.type == 'conv':
                            conv_count += 1
                        elif op.type == 'read_state':
                            read_state_count += 1
                        elif op.type == 'coreml_update_state':
                            update_state_count += 1
                        elif op.type == 'slice_update':
                            scatter_count += 1

            # Expected: 7 projections per layer (q,k,v,o,gate,up,down)
            expected_constexpr = num_layers * 7
            expected_matmul_scales = num_layers * 7  # A@B for each projection
            expected_slice_updates = num_layers * 2  # K and V update per layer
            # ANE pattern: read_state -> slice_update -> write_state -> read_state
            # 1 initial read + 2 reads per layer (after K and V)
            expected_read_state = 1 + num_layers * 2
            # 2 writes per layer (after K and V)
            expected_update_state = num_layers * 2

            print(f"  constexpr_lut_to_dense: {constexpr_count} (expected {expected_constexpr})")
            print(f"  matmul: {matmul_count} (includes A@B + attention)")
            print(f"  conv: {conv_count}")
            print(f"  read_state: {read_state_count} (expected {expected_read_state})")
            print(f"  coreml_update_state: {update_state_count} (expected {expected_update_state})")
            print(f"  slice_update: {scatter_count} (expected {expected_slice_updates})")

            if constexpr_count == expected_constexpr:
                print("  ✓ AQ1 structure verified: LUT + dynamic scales preserved!")
            else:
                print(f"  ⚠ LUT count mismatch")

            if read_state_count == expected_read_state and update_state_count == expected_update_state:
                print("  ✓ KV cache state verified (ANE pattern)!")
            else:
                print(f"  ⚠ KV cache state count mismatch")

        return mlmodel

    # =========================================================================
    # A1F Conversion: LUT-compressed snapped weights + factored scales (A @ B)
    # =========================================================================

    def convert_part_2_a1f(
        self, model: QwenForCausalLM, chunk_idx: int = 0, total_chunks: int = 1
    ) -> ct.models.MLModel:
        """Convert FFN layers using A1F approach.

        This method creates a model with:
        - Snapped weights compressed via constexpr_lut_to_dense
        - Factored scales (A @ B) computed at runtime via matmul
        - Full transformer layers (attention + MLP)

        Args:
            model: The Qwen model (used for config, weights come from checkpoint)
            chunk_idx: Which chunk to convert (0-indexed)
            total_chunks: Total number of chunks

        Returns:
            ct.models.MLModel: Converted CoreML model
        """
        require_coreml()

        if not self.aq1_checkpoint:
            raise ValueError("A1F conversion requires --aq1-checkpoint")

        config = model.model.config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", hidden_size // num_heads)
        total_layers = config.num_hidden_layers

        # Compute layer range for this chunk
        layers_per_chunk = total_layers // total_chunks
        start_layer = chunk_idx * layers_per_chunk
        end_layer = min((chunk_idx + 1) * layers_per_chunk, total_layers)
        if chunk_idx == total_chunks - 1:
            end_layer = total_layers  # Last chunk gets remaining layers

        print(f"\n[A1F] Converting layers {start_layer}-{end_layer-1} (chunk {chunk_idx+1}/{total_chunks})")
        print(f"      hidden_size={hidden_size}, intermediate={intermediate_size}")
        print(f"      num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

        # Load checkpoint data
        print(f"[A1F] Loading checkpoint: {self.aq1_checkpoint}")
        ckpt = torch.load(self.aq1_checkpoint, map_location='cpu', weights_only=False)

        # Build A1F wrapper model
        from tests.dev.test_qwen_a1f_layer import (
            QwenStackWithFactoredScales,
            QwenRMSNorm,
        )

        layers_data = []
        for layer_idx in range(start_layer, end_layer):
            print(f"[A1F] Loading layer {layer_idx}...")
            layer_data = {'mlp': {}, 'attn': {}, 'norm': {}}

            # MLP projections (2-bit)
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                prefix = f'model.layers.{layer_idx}.mlp.{proj}'
                layer_data['mlp'][proj] = {
                    'snapped': ckpt[f'{prefix}.weight'].to(torch.float16),
                    'scale_A': ckpt[f'{prefix}.scale_A'].to(torch.float16),
                    'scale_B': ckpt[f'{prefix}.scale_B'].to(torch.float16),
                    'lut': ckpt[f'{prefix}.lut'].to(torch.float16),
                }

            # Attention projections (4-bit)
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                prefix = f'model.layers.{layer_idx}.self_attn.{proj}'
                layer_data['attn'][proj] = {
                    'snapped': ckpt[f'{prefix}.weight'].to(torch.float16),
                    'scale_A': ckpt[f'{prefix}.scale_A'].to(torch.float16),
                    'scale_B': ckpt[f'{prefix}.scale_B'].to(torch.float16),
                    'lut': ckpt[f'{prefix}.lut'].to(torch.float16),
                }

            # Head norms
            layer_data['attn']['q_norm'] = ckpt.get(
                f'model.layers.{layer_idx}.self_attn.q_norm.weight',
                torch.ones(head_dim, dtype=torch.float16)
            ).to(torch.float16)
            layer_data['attn']['k_norm'] = ckpt.get(
                f'model.layers.{layer_idx}.self_attn.k_norm.weight',
                torch.ones(head_dim, dtype=torch.float16)
            ).to(torch.float16)

            # Layer norms
            layer_data['norm']['input_layernorm'] = ckpt[
                f'model.layers.{layer_idx}.input_layernorm.weight'
            ].to(torch.float16)
            layer_data['norm']['post_attention_layernorm'] = ckpt[
                f'model.layers.{layer_idx}.post_attention_layernorm.weight'
            ].to(torch.float16)

            layers_data.append(layer_data)

        # Create wrapper model
        print("[A1F] Creating model with factored scales...")
        wrapper = QwenStackWithFactoredScales(
            layers_data=layers_data,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            context_length=self.context_length,
        )
        wrapper.eval()
        wrapper = wrapper.half()

        # Trace
        print("[A1F] Tracing model...")
        x_sample = torch.randn(1, hidden_size, 1, 1, dtype=torch.float16)
        pos_sample = torch.tensor([0], dtype=torch.int32)
        mask_sample = torch.zeros(1, 1, 1, self.context_length, dtype=torch.float16)

        traced = torch.jit.trace(wrapper, (x_sample, pos_sample, mask_sample), strict=False)

        # Apply patch to prevent const folding
        original_value_inference = _patch_value_inference()

        try:
            # Convert with DEFAULT pipeline - const_elimination needed for scalar constants (epsilon)
            # The _patch_value_inference handles selective blocking of large tensor folding
            print("[A1F] Converting to CoreML...")

            mlmodel = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(name="hidden_states", shape=(1, hidden_size, 1, 1), dtype=np.float16),
                    ct.TensorType(name="position_ids", shape=(1,), dtype=np.int32),
                    ct.TensorType(name="causal_mask", shape=(1, 1, 1, self.context_length), dtype=np.float16),
                ],
                outputs=[ct.TensorType(name="output", dtype=np.float16)],
                convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.iOS18,
                # Note: Keep DEFAULT pipeline - const_elimination needed for scalar constants
            )
        finally:
            _restore_value_inference(original_value_inference)

        # Find snapped ops and apply selective palettization
        snapped_ops = _find_snapped_ops(mlmodel)
        print(f"[A1F] Found {len(snapped_ops)} snapped weight ops")

        if snapped_ops:
            mlmodel = _apply_selective_palettization(mlmodel, snapped_ops, mode="unique")

        # Verify structure
        print("\n[A1F] Verifying model structure...")
        mil_prog = mlmodel._mil_program
        if mil_prog:
            lut_count = 0
            matmul_count = 0
            for func in mil_prog.functions.values():
                for op in func.operations:
                    if op.op_type == 'constexpr_lut_to_dense':
                        lut_count += 1
                    elif op.op_type == 'matmul':
                        matmul_count += 1

            num_layers = end_layer - start_layer
            expected_lut = num_layers * 7  # 7 projections per layer
            expected_matmul = num_layers * 7  # 7 A@B per layer + attention matmuls

            print(f"  constexpr_lut_to_dense: {lut_count} (expected {expected_lut})")
            print(f"  matmul: {matmul_count} (includes A@B and attention)")

            if lut_count == expected_lut:
                print("  ✓ A1F structure verified: LUT + matmul preserved!")
            else:
                print(f"  ⚠ LUT count mismatch: got {lut_count}, expected {expected_lut}")

        self.converted_model = mlmodel
        return mlmodel

    def convert_part_2_a1f_prefill(
        self, model: QwenForCausalLM, chunk_idx: int = 0, total_chunks: int = 1
    ) -> ct.models.MLModel:
        """Convert FFN layers for prefill mode using A1F approach.

        Similar to convert_part_2_a1f but for batch processing.
        """
        # TODO: Implement prefill variant with batch_size input shape
        raise NotImplementedError("A1F prefill mode not yet implemented - use single-token mode")

    def convert_monolithic(
        self, model: QwenForCausalLM, is_prefill: bool = False
    ) -> ct.models.MLModel:
        """Convert full model (embeddings + FFN + LM head) to single CoreML model.

        This creates a monolithic model that takes input_ids and returns logits,
        combining all components into a single file for simpler deployment.

        Args:
            model: The Qwen model to convert
            is_prefill: If True, convert for prefill mode (batch processing)
                       If False, convert for inference mode (single token)

        Returns:
            ct.models.MLModel: Monolithic CoreML model
        """
        require_coreml()
        mode_str = "prefill" if is_prefill else "inference"
        print(f"\nConverting monolithic model for {mode_str} mode...")

        class MonolithicWrapper(torch.nn.Module):
            """Wrapper combining embeddings + transformer + LM head."""

            def __init__(
                self, model: QwenForCausalLM, context_length: int, is_prefill: bool
            ) -> None:
                super().__init__()
                self.model = model
                self.context_length = context_length
                self.is_prefill = is_prefill

                # Determine LM head mode
                if hasattr(model, "lm_head16_1"):
                    self.lm_head_mode = "16"
                    self.lm_heads = [
                        getattr(model, f"lm_head16_{i}") for i in range(1, 17)
                    ]
                elif hasattr(model, "lm_head8_1"):
                    self.lm_head_mode = "8"
                    self.lm_heads = [
                        getattr(model, f"lm_head8_{i}") for i in range(1, 9)
                    ]
                elif hasattr(model, "lm_head2_1"):
                    self.lm_head_mode = "2"
                    self.lm_heads = [model.lm_head2_1, model.lm_head2_2]
                elif hasattr(model, "lm_head1"):
                    self.lm_head_mode = "1"
                    self.lm_head = model.lm_head1
                else:
                    self.lm_head_mode = "linear"
                    self.lm_head = model.lm_head

            def forward(
                self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                causal_mask: torch.Tensor,
                current_pos: torch.Tensor,
            ) -> tuple:
                # Step 1: Embeddings
                hidden_states = self.model.model.embed_tokens(input_ids)
                hidden_states = hidden_states.to(MODEL_DTYPE)

                # Step 2: Transformer layers
                if self.is_prefill:
                    rotary = self.model.model.get_rotary_embedding_prefill(position_ids)
                else:
                    rotary = self.model.model.get_rotary_embeddings_s(current_pos)

                hidden_states = self.model.model.process_layers(
                    hidden_states,
                    position_ids,
                    causal_mask,
                    current_pos,
                    rotary,
                    start_layer=0,
                    end_layer=None,
                    IN_PREFILL=self.is_prefill,
                )

                # Apply final layer norm
                hidden_states = self.model.model.norm(hidden_states)

                # Step 3: LM Head
                if self.lm_head_mode != "linear":
                    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)

                if self.lm_head_mode == "16":
                    return tuple(
                        h(hidden_states).squeeze(2).transpose(1, 2)
                        for h in self.lm_heads
                    )
                elif self.lm_head_mode == "8":
                    return tuple(
                        h(hidden_states).squeeze(2).transpose(1, 2)
                        for h in self.lm_heads
                    )
                elif self.lm_head_mode == "2":
                    logits1 = self.lm_heads[0](hidden_states).squeeze(2).transpose(1, 2)
                    logits2 = self.lm_heads[1](hidden_states).squeeze(2).transpose(1, 2)
                    return logits1, logits2
                elif self.lm_head_mode == "1":
                    return (self.lm_head(hidden_states).squeeze(2).transpose(1, 2),)
                else:
                    return (self.lm_head(hidden_states),)

        wrapper = MonolithicWrapper(model, self.context_length, is_prefill)
        wrapper.eval()

        # Ensure no gradients
        for param in wrapper.parameters():
            param.requires_grad = False

        print(f"Monolithic wrapper created (LM head mode: {wrapper.lm_head_mode})")

        # Prepare inputs based on mode
        if is_prefill:
            # Prefill mode: batch processing
            sample_input_ids = torch.zeros(
                (1, self.batch_size), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_position_ids = torch.zeros(
                (self.batch_size,), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_causal_mask = torch.zeros(
                (1, 1, self.batch_size, self.context_length),
                dtype=torch.float16,
                device=TEST_DEVICE,
            )
        else:
            # Inference mode: single token
            sample_input_ids = torch.zeros(
                (1, 1), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_position_ids = torch.zeros(
                (1,), dtype=torch.int32, device=TEST_DEVICE
            )
            sample_causal_mask = torch.zeros(
                (1, 1, 1, self.context_length),
                dtype=torch.float16,
                device=TEST_DEVICE,
            )

        sample_current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        print(f"Sample inputs ({mode_str} mode):")
        print(f"  input_ids: {sample_input_ids.shape}")
        print(f"  position_ids: {sample_position_ids.shape}")
        print(f"  causal_mask: {sample_causal_mask.shape}")
        print(f"  current_pos: {sample_current_pos.shape}")

        # Trace model
        print("Tracing monolithic model...")
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper,
                (
                    sample_input_ids,
                    sample_position_ids,
                    sample_causal_mask,
                    sample_current_pos,
                ),
            )
        print("Tracing completed!")

        # Define outputs based on LM head mode
        if wrapper.lm_head_mode == "16":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16)
                for i in range(1, 17)
            ]
        elif wrapper.lm_head_mode == "8":
            outputs = [
                ct.TensorType(name=f"logits{i}", dtype=np.float16)
                for i in range(1, 9)
            ]
        elif wrapper.lm_head_mode == "2":
            outputs = [
                ct.TensorType(name="logits1", dtype=np.float16),
                ct.TensorType(name="logits2", dtype=np.float16),
            ]
        else:
            outputs = [ct.TensorType(name="logits", dtype=np.float16)]

        # Convert to CoreML
        print("Starting CoreML conversion...")
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="input_ids", shape=sample_input_ids.shape, dtype=np.int32
                ),
                ct.TensorType(
                    name="position_ids", shape=sample_position_ids.shape, dtype=np.int32
                ),
                ct.TensorType(
                    name="causal_mask", shape=sample_causal_mask.shape, dtype=np.float16
                ),
                ct.TensorType(
                    name="current_pos", shape=sample_current_pos.shape, dtype=np.int32
                ),
            ],
            outputs=outputs,
            states=self.GetTransformerStates(model, part=None, prefix="model.model."),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        print(f"CoreML conversion for monolithic {mode_str} completed!")

        # Apply LUT quantization if specified
        if self.lut_bits:
            print(f"Applying LUT quantization ({self.lut_bits} bits)...")
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)
            mlmodel = self.converted_model

        return mlmodel


def parse_lut_arg(lut_value):
    """Parse LUT argument that can be either 'bits' or 'bits,per_channel'.

    Args:
        lut_value: String value from command line (e.g., '6' or '6,4')

    Returns:
        tuple: (lut_bits, per_channel) where per_channel defaults to 8 if not specified
    """
    if lut_value is None:
        return None, 8

    if ',' in lut_value:
        parts = lut_value.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid LUT format: {lut_value}. Expected format: 'bits' or 'bits,per_channel'")
        try:
            lut_bits = int(parts[0])
            per_channel = int(parts[1])
            return lut_bits, per_channel
        except ValueError:
            raise ValueError(f"Invalid LUT format: {lut_value}. Both values must be integers")
    else:
        try:
            lut_bits = int(lut_value)
            return lut_bits, 8  # Default per_channel value
        except ValueError:
            raise ValueError(f"Invalid LUT bits value: {lut_value}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the converter."""

    parser = argparse.ArgumentParser(description="Convert Qwen model to CoreML format")

    parser.add_argument(
        "--model",
        type=str,
        help="Path to model directory (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="qwen",
        help="Prefix for output filenames",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prefill",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=CONTEXT_LENGTH,
        help="Maximum context length",
    )
    parser.add_argument(
        "--lut",
        type=str,
        default=None,
        help='Use LUT quantization with N bits, optionally specify per_channel as "bits,per_channel" (e.g., "6,4"). Default per_channel is 8',
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Split FFN/prefill into N chunks",
    )
    parser.add_argument(
        "--part",
        type=str,
        choices=["1", "2", "2_prefill", "3", "all", "full", "prefill", "embeddings", "monolithic", "monolithic_prefill", "aq1_mlp", "aq1_ffn", "aq1_full", "a1f"],
        default="all",
        help="Model part to convert (aq1_mlp/aq1_ffn = MLP only, aq1_full = full transformer with attention+KV cache, a1f = trace-based AQ1)",
    )
    parser.add_argument(
        "--aq1-checkpoint",
        type=str,
        default=None,
        help="Path to ANEMLL checkpoint with snapped weights + scales for AQ1 conversion. "
             "Required when using --part aq1_mlp.",
    )
    parser.add_argument(
        "--aq1-nbits-mlp",
        type=int,
        default=2,
        help="Number of bits for MLP LUT quantization in AQ1 (default: 2)",
    )
    parser.add_argument(
        "--aq1-nbits-attn",
        type=int,
        default=4,
        help="Number of bits for attention LUT quantization in AQ1 (default: 4)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Limit number of transformer layers to convert (for quick testing). "
             "If not specified, converts all layers from model config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory for converted models",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to ANEMLL checkpoint with baked weights (snapped + scale_A + scale_B). "
             "If provided, weights are loaded from checkpoint instead of HuggingFace model.",
    )

    return parser.parse_args()


def test_conversion(
    model: Optional[QwenForCausalLM] = None,
    model_path: Optional[str] = None,
    prefix: str = "qwen",
    context_length: int = CONTEXT_LENGTH,
    lut_bits: Optional[int] = None,
    batch_size: int = 64,
    output_dir: str = ".",
    part: str = "full",
    num_chunks: int = 1,
    per_channel: int = 8,
    checkpoint_path: Optional[str] = None,
    aq1_checkpoint: Optional[str] = None,
    aq1_nbits_mlp: int = 2,
    aq1_nbits_attn: int = 4,
    num_layers: Optional[int] = None,
) -> ct.models.MLModel | List[ct.models.MLModel]:
    """Convert a Qwen model and save the result.

    Args:
        model: Pre-loaded Qwen model (optional)
        model_path: Path to model directory
        prefix: Model name prefix
        context_length: Context length for conversion
        lut_bits: LUT quantization bits
        batch_size: Batch size for conversion
        output_dir: Output directory
        part: Part to convert ("full" or "prefill")
        per_channel: Group size for per_grouped_channel quantization
        checkpoint_path: Path to ANEMLL checkpoint with baked weights (optional)
    """
    print(
        f"test_conversion called with model_path={model_path}, prefix={prefix}, part={part}"
    )
    if checkpoint_path:
        print(f"Using ANEMLL checkpoint: {checkpoint_path}")

    if model is None:
        if model_path is None:
            raise ValueError("model_path must be provided if model is None")

        config_path = os.path.join(model_path, "config.json")
        print(f"Looking for config at: {config_path}")
        if not os.path.exists(config_path):
            # Try to download from HuggingFace if it's a model ID
            print(f"Config not found locally, trying to download from HuggingFace: {model_path}")
            try:
                from huggingface_hub import snapshot_download
                local_path = snapshot_download(model_path, local_files_only=False)
                config_path = os.path.join(local_path, "config.json")
                model_path = local_path  # Update model_path for weight loading
                print(f"Downloaded to: {local_path}")
            except Exception as e:
                raise ValueError(f"Config file not found at {config_path} and failed to download from HuggingFace: {e}")

        print("Loading config...")
        config = QwenConfig.from_json(config_path)
        print(
            f"Config loaded: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}"
        )

        # Update config to match conversion parameters
        config.context_length = context_length
        config.state_length = max(
            config.state_length, context_length
        )  # Ensure state_length is at least context_length

        # Override number of layers if specified (for quick testing)
        if num_layers is not None:
            original_layers = config.num_hidden_layers
            config.num_hidden_layers = min(num_layers, original_layers)
            print(f"Limiting layers: {original_layers} -> {config.num_hidden_layers}")

        print(
            f"Updated config: context_length={config.context_length}, state_length={config.state_length}, num_layers={config.num_hidden_layers}"
        )

        print("Creating model...")
        model = QwenForCausalLM(config, enable_coreml=True)

        # Load weights from checkpoint or HuggingFace
        if checkpoint_path:
            print("Loading DYNAMIC A*B weights from ANEMLL checkpoint...")
            from anemll.models.anemll_quant import load_dynamic_weights_for_ane
            success = load_dynamic_weights_for_ane(model, checkpoint_path, verbose=True)
            if not success:
                raise ValueError(f"Failed to load weights from checkpoint: {checkpoint_path}")
            print("ANEMLL checkpoint with dynamic A*B loaded successfully!")
        else:
            print("Loading pretrained weights from HuggingFace...")
            model.load_pretrained_weights(model_path)
            print("Model loaded successfully!")
        
        # Ensure model is in eval mode and gradients are disabled
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("Model set to eval mode and gradients disabled")

    print("Creating converter...")
    converter = QwenConverter(
        model=model,
        context_length=context_length,
        batch_size=batch_size,
        lut_bits=lut_bits,
        num_chunks=num_chunks,
        per_channel=per_channel,
        aq1_checkpoint=aq1_checkpoint,
        aq1_nbits_mlp=aq1_nbits_mlp,
        aq1_nbits_attn=aq1_nbits_attn,
        num_layers=num_layers,  # Pass num_layers for AQ1 layer filtering
    )

    print("Starting conversion...")
    mlmodel = converter.convert(part=part)
    print("Conversion completed!")

    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(mlmodel, list):
        models = mlmodel
    else:
        models = [mlmodel]

    for i, m in enumerate(models):
        AddMetadata(
            m,
            {
                "context_length": context_length,
                "batch_size": batch_size if part in ["2_prefill", "prefill", "monolithic_prefill"] else None,
                "lut_bits": lut_bits,
                "num_chunks": num_chunks if part in ["2", "2_prefill"] else None,
                "chunk_no": i + 1 if part in ["2", "2_prefill"] else None,
                "split_part": (
                    ModelPart.FULL.value if part in ["full", "all", "123"] else part
                ),
            },
        )
        fname = f"{prefix}"
        if part in ["1", "embeddings"]:
            fname += "_embeddings"
        elif part in ["3"]:
            fname += "_lm_head"
        elif part == "monolithic":
            fname += "_monolithic"
        elif part == "monolithic_prefill":
            fname += "_monolithic_prefill"
        elif part == "aq1_mlp":
            fname += "_aq1_mlp"
            fname += f"_{aq1_nbits_mlp}bit"
            if num_chunks > 1:
                fname += f"_chunk_{i+1:02d}of{num_chunks:02d}"
        elif part == "aq1_ffn":
            fname += "_aq1_ffn"
            fname += f"_{aq1_nbits_mlp}bit"
            if num_chunks > 1:
                fname += f"_chunk_{i+1:02d}of{num_chunks:02d}"
        elif part == "aq1_full":
            fname += "_aq1_full"
            fname += f"_{aq1_nbits_mlp}bit"
            if num_chunks > 1:
                fname += f"_chunk_{i+1:02d}of{num_chunks:02d}"
        elif part in ["2", "2_prefill"]:
            base = "FFN" if part == "2" else "prefill"
            fname += f"_{base}"
            if lut_bits is not None:
                fname += f"_lut{lut_bits}"
            fname += f"_chunk_{i+1:02d}of{num_chunks:02d}"
        if part in ["full", "all", "123"]:
            fname += ""
        if part not in ["2", "2_prefill", "aq1_mlp", "aq1_ffn", "aq1_full"]:
            if lut_bits is not None:
                fname += f"_lut{lut_bits}"
            fname += ".mlpackage"
        else:
            fname += ".mlpackage"
        out_path = os.path.join(output_dir, fname)
        print(f"Saving model to: {out_path}")
        m.save(out_path)

    return mlmodel


def main() -> None:
    print("Starting qwen_converter main()...")
    args = parse_args()
    print(f"Parsed args: {args}")

    # Parse LUT argument
    lut_bits, per_channel = parse_lut_arg(args.lut)

    model_path = args.model if args.model else "Qwen/Qwen3-0.6B"

    # Get AQ1 args (use getattr because of hyphen in name)
    aq1_checkpoint = getattr(args, 'aq1_checkpoint', None)
    aq1_nbits_mlp = getattr(args, 'aq1_nbits_mlp', 2)
    aq1_nbits_attn = getattr(args, 'aq1_nbits_attn', 4)
    num_layers = getattr(args, 'num_layers', None)

    print(f"\nConverting model from: {model_path}")
    print(f"Output filename prefix: {args.prefix}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    if args.checkpoint:
        print(f"ANEMLL checkpoint: {args.checkpoint}")
        print("  (Weights will be baked from snapped * scale_A @ scale_B)")
    if aq1_checkpoint:
        print(f"AQ1 checkpoint: {aq1_checkpoint}")
        print(f"  MLP bits: {aq1_nbits_mlp}, Attention bits: {aq1_nbits_attn}")
        print("  (Using constexpr_lut_to_dense + dynamic A*B)")
    if lut_bits:
        print(f"LUT quantization: {lut_bits} bits, per_channel group size: {per_channel}")
    if args.chunk:
        print(f"Splitting into {args.chunk} chunks")
    if num_layers:
        print(f"Limiting to {num_layers} transformer layers")
    print(f"Converting part(s): {args.part}")

    # Map legacy part names to numeric equivalents
    part_map = {"full": "all", "embeddings": "1", "prefill": "2_prefill"}
    part = part_map.get(args.part, args.part)

    try:
        print("\nCalling test_conversion()...")
        result = test_conversion(
            model_path=model_path,
            prefix=args.prefix,
            context_length=args.context_length,
            lut_bits=lut_bits,
            batch_size=args.batch_size,
            output_dir=args.output,
            part=part,
            num_chunks=args.chunk or 1,
            per_channel=per_channel,
            checkpoint_path=args.checkpoint,
            aq1_checkpoint=aq1_checkpoint,
            aq1_nbits_mlp=aq1_nbits_mlp,
            aq1_nbits_attn=aq1_nbits_attn,
            num_layers=num_layers,
        )
        print(f"Conversion completed successfully! Result: {type(result)}")
    except Exception as e:  # pragma: no cover - CLI tool
        print(f"\nError during conversion: {str(e)}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
