"""
Sequential DataLoader for Video Frame Sequences

This module provides a DatasetSegmentation class that loads images
in sequential order based on video ID and frame ID, which is required
for temporal models like LRU.
"""

import json
import torch
import glob
import os
import re
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms

from src.processor import Samprocessor
import src.utils as utils


class DatasetSegmentationSequential(Dataset):
    """
    Dataset that loads images in sequential order by video ID and frame ID.

    This is crucial for temporal models that need to process video frames
    in chronological order.

    Arguments:
        config_file (dict): Configuration dictionary
        processor (Samprocessor): SAM processor for image preprocessing
        mode (str): Dataset mode ('train', 'val', 'test')

    Returns:
        dict: Dictionary with keys (image, original_size, boxes, ground_truth_mask, video_id, frame_id)
    """

    def __init__(self, config_file: dict, processor: Samprocessor, mode: str):
        super().__init__()

        self.processor = processor
        self.mode = mode

        # Storage for data
        self.data_entries = []  # List of (img_path, mask_path, bbox, video_id, frame_id)

        # Load data based on mode
        if "train" in mode:
            json_path = f"/mnt/hdd2/task2/sam_lora/output_bbox_{mode}.json"
            print(f"train set: {json_path}")
            with open(json_path, 'r') as f:
                all_data = json.load(f)

            base_path = config_file["DATASET"]["TRAIN_PATH"]
            mode_data = all_data.get(mode, {})

            for img_name, info_list in mode_data.items():
                img_path = os.path.join(base_path, 'images', img_name)

                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found - {img_path}")
                    continue

                # Extract video_id and frame_id from filename
                # Expected format: video{id}_frame{frame_id}.png or similar
                video_id, frame_id = self._extract_video_frame_id(img_name)

                for info in info_list:
                    self.data_entries.append({
                        'img_path': img_path,
                        'mask_path': info["mask_path"],
                        'bbox': info["bbox"],
                        'video_id': video_id,
                        'frame_id': frame_id
                    })

        elif "val" in mode:
            json_path = f"/mnt/hdd2/task2/sam_lora/output_bbox_{mode}.json"
            print(f"val set: {json_path}")
            with open(json_path, 'r') as f:
                all_data = json.load(f)

            base_path_val = "/mnt/hdd2/task2/sam_lora/train"
            mode_data = all_data.get(mode, {})

            for img_name, info_list in mode_data.items():
                img_path = os.path.join(base_path_val, 'images', img_name)

                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found - {img_path}")
                    continue

                video_id, frame_id = self._extract_video_frame_id(img_name)

                for info in info_list:
                    self.data_entries.append({
                        'img_path': img_path,
                        'mask_path': info["mask_path"],
                        'bbox': info["bbox"],
                        'video_id': video_id,
                        'frame_id': frame_id
                    })

        else:  # test mode
            json_path = "/mnt/hdd2/task2/sam_lora/output_bbox_test_19.json"
            with open(json_path, 'r') as f:
                all_data = json.load(f)

            base_path = config_file["DATASET"]["TEST_PATH"]
            mode_data = all_data.get(mode, {})

            for img_name, info_list in mode_data.items():
                img_path = os.path.join(base_path, 'images', img_name)

                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found - {img_path}")
                    continue

                video_id, frame_id = self._extract_video_frame_id(img_name)

                for info in info_list:
                    self.data_entries.append({
                        'img_path': img_path,
                        'mask_path': info["mask_path"],
                        'bbox': info["bbox"],
                        'video_id': video_id,
                        'frame_id': frame_id
                    })

        # Sort data by video_id and frame_id to ensure sequential order
        self.data_entries.sort(key=lambda x: (x['video_id'], x['frame_id']))

        print(f"Loaded {len(self.data_entries)} samples in sequential order")
        if len(self.data_entries) > 0:
            print(f"Video ID range: {self.data_entries[0]['video_id']} to {self.data_entries[-1]['video_id']}")
            print(f"First entry: video_{self.data_entries[0]['video_id']}_frame_{self.data_entries[0]['frame_id']}")
            print(f"Last entry: video_{self.data_entries[-1]['video_id']}_frame_{self.data_entries[-1]['frame_id']}")

    def _extract_video_frame_id(self, filename: str):
        """
        Extract video ID and frame ID from filename.

        Supports multiple naming patterns:
        - video{id}_frame{frame}.png
        - {video_id}_{frame_id}.png
        - frame_{frame_id}_video_{video_id}.png

        Returns:
            tuple: (video_id, frame_id) both as integers
        """
        # Remove extension
        basename = os.path.splitext(filename)[0]

        # Try pattern: video{id}_frame{frame}
        match = re.search(r'video[_-]?(\d+).*?frame[_-]?(\d+)', basename, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))

        # Try pattern: frame{frame}_video{id}
        match = re.search(r'frame[_-]?(\d+).*?video[_-]?(\d+)', basename, re.IGNORECASE)
        if match:
            return int(match.group(2)), int(match.group(1))

        # Try pattern: {video_id}_{frame_id} (generic)
        parts = basename.split('_')
        if len(parts) >= 2:
            # Try to extract two numbers
            numbers = [int(p) for p in parts if p.isdigit()]
            if len(numbers) >= 2:
                return numbers[0], numbers[1]

        # Fallback: try to find any two numbers in the filename
        numbers = re.findall(r'\d+', basename)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        elif len(numbers) == 1:
            # If only one number, use it as frame_id and video_id=0
            return 0, int(numbers[0])
        else:
            # No numbers found, use hash of filename
            print(f"Warning: Could not extract video/frame ID from {filename}, using hash")
            return hash(filename) % 10000, 0

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, index: int) -> dict:
        """
        Get item by index.

        Returns a dictionary containing processed image, mask, and metadata.
        """
        entry = self.data_entries[index]

        # Load image and mask
        image = Image.open(entry['img_path'])
        mask = Image.open(entry['mask_path'])
        mask = mask.convert('1')
        ground_truth_mask = np.array(mask)
        original_size = tuple(image.size)[::-1]

        # Get bounding box
        box = utils.get_bounding_box(ground_truth_mask)

        # Process with SAM processor
        inputs = self.processor(image, original_size, box)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)

        # Add metadata for sequence tracking
        inputs["video_id"] = entry['video_id']
        inputs["frame_id"] = entry['frame_id']
        inputs["img_path"] = entry['img_path']

        return inputs

    def get_video_sequences(self):
        """
        Get information about video sequences in the dataset.

        Returns:
            dict: Dictionary mapping video_id to list of frame indices
        """
        video_sequences = {}
        for idx, entry in enumerate(self.data_entries):
            video_id = entry['video_id']
            if video_id not in video_sequences:
                video_sequences[video_id] = []
            video_sequences[video_id].append(idx)

        return video_sequences


def collate_fn_sequential(batch: list) -> list:
    """
    Collate function for sequential dataloader.

    This is the same as the original collate_fn, but we keep it here
    for clarity and potential future modifications.

    Arguments:
        batch: List of samples from the dataset

    Returns:
        list: List of dictionaries (preserves batch structure)
    """
    return list(batch)


class SequentialSampler(torch.utils.data.Sampler):
    """
    Sampler that ensures frames are sampled in sequential order.

    This is essentially the same as SequentialSampler but included
    here for explicitness about the sequential requirement.
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
