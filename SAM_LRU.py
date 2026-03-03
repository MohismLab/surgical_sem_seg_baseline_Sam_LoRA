"""
SAM with LoRA and LRU Integration

This module integrates:
1. SAM (Segment Anything Model) - Base segmentation model
2. LoRA (Low-Rank Adaptation) - Efficient fine-tuning
3. LRU (Linear Recurrent Unit) - Temporal sequence modeling

The LRU layer is inserted between SAM's image encoder and mask decoder
to capture temporal dependencies across video frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional

from src.segment_anything.modeling.sam import Sam
from src.lora import LoRA_sam
from LRU import LRU, LRUBlock


class LRU_SAM(nn.Module):
    """
    SAM model with LoRA and LRU integration.

    Architecture:
        Input Images -> SAM Image Encoder (with LoRA) -> LRU Layer ->
        SAM Prompt Encoder + Mask Decoder -> Output Masks

    The LRU layer processes the spatial features as temporal sequences,
    enabling the model to leverage information from previous frames.

    Arguments:
        sam_model: Pre-trained SAM model
        lora_rank: Rank for LoRA adaptation
        lru_hidden_dim: Hidden dimension for LRU
        use_lru: Whether to use LRU (can be disabled for baseline comparison)
        lora_layer: List of layers to apply LoRA (None for all layers)
    """

    def __init__(
        self,
        sam_model: Sam,
        lora_rank: int = 2,
        lru_hidden_dim: int = 512,
        use_lru: bool = True,
        lora_layer: Optional[List[int]] = None,
    ):
        super().__init__()

        # Apply LoRA to SAM
        self.lora_sam = LoRA_sam(sam_model, lora_rank, lora_layer)
        self.sam = self.lora_sam.sam

        # Get feature dimensions from SAM image encoder
        # For ViT-B: embed_dim=768, after neck: out_chans=256
        # Output shape: (B, 256, 64, 64) for 1024x1024 input
        self.feature_channels = self.sam.image_encoder.neck[0].out_channels  # 256
        self.feature_spatial_size = 64  # 1024 / 16 = 64

        # Calculate sequence length (flatten spatial dimensions)
        self.seq_len = self.feature_spatial_size * self.feature_spatial_size  # 4096

        # LRU layer for temporal modeling
        self.use_lru = use_lru
        if self.use_lru:
            self.lru = LRUBlock(
                d_model=self.feature_channels,
                d_hidden=lru_hidden_dim,
                ffn_dim=self.feature_channels * 2,
                dropout=0.1
            )
            print(f"LRU initialized: d_model={self.feature_channels}, "
                  f"d_hidden={lru_hidden_dim}, seq_len={self.seq_len}")
        else:
            self.lru = None
            print("LRU disabled - running baseline SAM-LoRA")

        # Hidden state for recurrent processing
        self.hidden_state = None

    def reset_hidden_state(self):
        """Reset the hidden state (call at the start of a new video sequence)."""
        self.hidden_state = None

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        reset_state: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass with temporal modeling.

        Arguments:
            batched_input: List of input dictionaries (same as SAM)
            multimask_output: Whether to output multiple masks
            reset_state: Whether to reset the hidden state before processing

        Returns:
            List of output dictionaries with masks and predictions
        """
        if reset_state:
            self.reset_hidden_state()

        # Preprocess images
        input_images = torch.stack(
            [self.sam.preprocess(x["image"].squeeze(0)) for x in batched_input],
            dim=0
        )
        batch_size = input_images.shape[0]

        # Extract image features with SAM encoder (with LoRA)
        image_embeddings = self.sam.image_encoder(input_images)
        # Shape: (B, C, H, W) e.g., (B, 256, 64, 64)

        # Apply LRU if enabled
        if self.use_lru:
            image_embeddings = self._apply_lru(image_embeddings)

        # Process each image with prompt encoder and mask decoder
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # Get prompts
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None

            # Encode prompts
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )

            # Decode masks
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            # Postprocess masks
            masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )

            low_res_mask_reshaped = masks
            masks = masks > self.sam.mask_threshold

            outputs.append({
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_mask_reshaped,
            })

        return outputs

    def _apply_lru(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply LRU to image features.

        Transforms spatial features into sequences, applies LRU,
        then transforms back to spatial format.

        Arguments:
            features: Image features from encoder (B, C, H, W)

        Returns:
            Processed features (B, C, H, W)
        """
        B, C, H, W = features.shape

        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        # This treats each spatial location as a time step
        features_flat = features.flatten(2).permute(0, 2, 1)
        # Shape: (B, seq_len, C) where seq_len = H*W

        # Apply LRU
        features_lru, self.hidden_state = self.lru(
            features_flat,
            state=self.hidden_state
        )
        # Shape: (B, seq_len, C)

        # Reshape back to spatial format: (B, seq_len, C) -> (B, C, H, W)
        features_out = features_lru.permute(0, 2, 1).reshape(B, C, H, W)

        return features_out

    def get_lora_parameters(self):
        """Get LoRA parameters for optimization."""
        return self.lora_sam.A_weights + self.lora_sam.B_weights

    def get_lru_parameters(self):
        """Get LRU parameters for optimization."""
        if self.lru is not None:
            return list(self.lru.parameters())
        return []

    def get_trainable_parameters(self):
        """Get all trainable parameters (LoRA + LRU)."""
        params = []

        # LoRA parameters
        for weight in self.lora_sam.A_weights + self.lora_sam.B_weights:
            params.extend(weight.parameters())

        # LRU parameters
        if self.use_lru and self.lru is not None:
            params.extend(self.lru.parameters())

        return params

    def save_lora_parameters(self, filename: str):
        """Save LoRA parameters."""
        self.lora_sam.save_lora_parameters(filename)

    def load_lora_parameters(self, filename: str):
        """Load LoRA parameters."""
        self.lora_sam.load_lora_parameters(filename)

    def save_checkpoint(self, filename: str):
        """
        Save complete checkpoint including LoRA and LRU weights.

        Arguments:
            filename: Path to save checkpoint
        """
        checkpoint = {
            'lru_state_dict': self.lru.state_dict() if self.lru is not None else None,
            'hidden_state': self.hidden_state,
            'config': {
                'feature_channels': self.feature_channels,
                'feature_spatial_size': self.feature_spatial_size,
                'use_lru': self.use_lru,
            }
        }

        # Save LoRA separately (using existing method)
        lora_filename = filename.replace('.pth', '_lora.safetensors')
        self.save_lora_parameters(lora_filename)

        # Save LRU and other state
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
        print(f"LoRA weights saved: {lora_filename}")

    def load_checkpoint(self, filename: str, lora_filename: Optional[str] = None):
        """
        Load complete checkpoint including LoRA and LRU weights.

        Arguments:
            filename: Path to checkpoint file
            lora_filename: Path to LoRA weights (if None, inferred from filename)
        """
        # Load LRU and state
        checkpoint = torch.load(filename, map_location=self.sam.device)

        if self.lru is not None and checkpoint['lru_state_dict'] is not None:
            self.lru.load_state_dict(checkpoint['lru_state_dict'])
            print(f"LRU weights loaded from {filename}")

        self.hidden_state = checkpoint.get('hidden_state', None)

        # Load LoRA weights
        if lora_filename is None:
            lora_filename = filename.replace('.pth', '_lora.safetensors')

        self.load_lora_parameters(lora_filename)
        print(f"LoRA weights loaded from {lora_filename}")
