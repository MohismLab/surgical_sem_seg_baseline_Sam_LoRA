import json
import torch
import glob
import os 
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import src.utils as utils
import yaml

"""
改成从json读取train, val, test set
image:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 截取图片
        image = image[:1004, 289:1630 + 1, :]  # [y,x], 去除两侧黑边、底部噪声
nrrd:        
        nrrd_data, _ = nrrd.read(nrrd_path)
        # nrrd_data.shape: width, height, depth
        # 裁剪数据,提取二维切片
        # T: 调整维度顺序width, height, depth为(H, W, C)
        nrrd_label = nrrd_data[289:1630 + 1, :1004, 0].squeeze().T.astype(np.uint8)
"""

class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks

    Arguments:
        folder_path (str): The path of the folder containing the images
        processor (obj): Samprocessor class that helps pre processing the image, and prompt 
    
    Return:
        (dict): Dictionnary with 4 keys (image, original_size, boxes, ground_truth_mask)
            image: image pre processed to 1024x1024 size
            original_size: Original size of the image before pre processing
            boxes: bouding box after adapting the coordinates of the pre processed image
            ground_truth_mask: Ground truth mask
    """

    def __init__(self, config_file: dict, processor: Samprocessor, mode: str):
        super().__init__()
        
        # if mode == "train":
        #     self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TRAIN_PATH"],'images','*.png'))
        #     self.mask_files = []
        #     for img_path in self.img_files:
        #         self.mask_files.append(os.path.join(config_file["DATASET"]["TRAIN_PATH"],'masks', os.path.basename(img_path)[:-4] + ".png")) 

        # else:
        #     self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TEST_PATH"],'images','*.png'))
        #     self.mask_files = []
        #     for img_path in self.img_files:
        #         self.mask_files.append(os.path.join(config_file["DATASET"]["TEST_PATH"],'masks', os.path.basename(img_path)[:-4] + ".png"))


        # 改成从json获取
        # train_set
        if "train" in mode: # ["train1", "train2", "train3", "train4", "train5"]
            json_path = f"/mnt/hdd2/task2/sam_lora/output_bbox_{mode}.json"
            print(f"train set: {json_path}")
            # 从JSON文件加载数据
            with open(json_path, 'r') as f:
                all_data = json.load(f)
            
            # 根据模式选择数据集
            self.img_files = []
            self.mask_files = []
            self.bboxes = []
            
            # 获取基础路径
            # TRAIN_PATH: "/mnt/hdd2/tasks/sam_lora/train"
            base_path = config_file["DATASET"]["TRAIN_PATH"]
            # 从JSON中提取当前模式的数据
            mode_data = all_data.get(mode, {})
            
            for img_name, info_list in mode_data.items():
                # 构建完整图片路径
                # print(f"img_name: {img_name.split('_')[0]}")
                img_path = os.path.join(base_path, 'images', img_name)
                
                # 验证文件是否存在
                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found - {img_path}")
                    continue
                
                # self.img_files.append(img_path)
                # self.mask_files.append(info["mask_path"])
                # self.bboxes.append(info["bbox"])     
                # 对每个图片的多个标注分别处理
                for info in info_list:
                    self.img_files.append(img_path)
                    self.mask_files.append(info["mask_path"])
                    self.bboxes.append(info["bbox"])       
            # print(f"imgs: {self.img_files}")
            # print(f"masks: {self.mask_files}")
        elif "val" in mode: # ["val1", "val2", "val3", "val4", "val5"]
            # val_set
            json_path = f"/mnt/hdd2/task2/sam_lora/output_bbox_{mode}.json"
            print(f"val set: {json_path}")
            # 从JSON文件加载数据
            with open(json_path, 'r') as f:
                all_data = json.load(f)
            
            # 根据模式选择数据集
            self.img_files = []
            self.mask_files = []
            self.bboxes = []
            
            # 获取基础路径
            base_path_val = "/mnt/hdd2/task2/sam_lora/train"
            # 从JSON中提取当前模式的数据
            mode_data = all_data.get(mode, {})
            
            for img_name, info_list in mode_data.items():
                # 构建完整图片路径
                # print(f"img_name: {img_name.split('_')[0]}")
                img_path = os.path.join(base_path_val, 'images', img_name)               
                
                # 验证文件是否存在
                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found - {img_path}")
                    continue
                
                # self.img_files.append(img_path)
                # self.mask_files.append(info["mask_path"])
                # self.bboxes.append(info["bbox"])     
                # 对每个图片的多个标注分别处理
                for info in info_list:
                    self.img_files.append(img_path)
                    self.mask_files.append(info["mask_path"])
                    self.bboxes.append(info["bbox"])       
            # print(f"imgs: {self.img_files}")
            # print(f"masks: {self.mask_files}")
        else:
            # test set
            # 从JSON文件加载数据
            json_path = "/mnt/hdd2/task2/sam_lora/output_bbox_test_19.json"
            with open(json_path, 'r') as f:
                all_data = json.load(f)
            
            # 根据模式选择数据集
            self.img_files = []
            self.mask_files = []
            self.bboxes = []
            
            # 获取基础路径
            # TRAIN_PATH: "/mnt/hdd2/tasks/sam_lora/test"
            base_path = config_file["DATASET"]["TEST_PATH"]
            
            # 从JSON中提取当前模式的数据
            mode_data = all_data.get(mode, {})
            
            for img_name, info_list in mode_data.items():
                # 构建完整图片路径
                img_path = os.path.join(base_path, 'images', img_name)
                
                # 验证文件是否存在
                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found - {img_path}")
                    continue
                
                for info in info_list:
                    self.img_files.append(img_path)
                    self.mask_files.append(info["mask_path"])
                    self.bboxes.append(info["bbox"])  


        self.processor = processor

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index: int) -> list:
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            # get image and mask in PIL format
            image =  Image.open(img_path)
            mask = Image.open(mask_path)
            mask = mask.convert('1')
            ground_truth_mask =  np.array(mask)
            original_size = tuple(image.size)[::-1]
    
            # get bounding box prompt
            box = utils.get_bounding_box(ground_truth_mask)
            inputs = self.processor(image, original_size, box)
            inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)

            return inputs
    
def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset
    
    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)