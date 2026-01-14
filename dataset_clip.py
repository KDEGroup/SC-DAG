import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class PairedImageDataset(Dataset):
    def __init__(self, src_dir, ref_results_file, target_dir, transform=None):
        """
        Args:
            src_dir (str): 存放源图像的目录
            ref_results_file (str): 参考选择结果文件(reference_selection_results.txt)
            target_dir (str): 存放目标图像的目录
            transform (callable, optional): 可选的图像变换
        """
        self.src_dir = src_dir
        self.target_dir = target_dir
        self.transform = transform
        # 读取参考选择结果
        self.ref_df = pd.read_csv(ref_results_file)
        #print(self.ref_df)
        
        # 获取所有源图像文件名 (cold_id.jpg)
        self.src_images = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
        
        # 创建cold_id到ref_id的映射字典
        self.pairing_dict = {}
        for _, row in self.ref_df.iterrows():
            cold_id = int(row['cold_id'])
            ref_id = int(row['ref_id'])
            self.pairing_dict[cold_id ] = ref_id 
        #print(self.pairing_dict)
        
    def __len__(self):
        return len(self.src_images)
    
    def __getitem__(self, index):
        # 获取源图像路径和对应的目标图像文件名
        src_img_name = self.src_images[index]
        cold_id = int(src_img_name.split('.')[0])  # 确保转换为整数
        
        # 查找对应的目标图像文件名
        tgt_img_name = self.pairing_dict.get(cold_id, None)
        if tgt_img_name is None:
            raise ValueError(f"No target image found for source image: {src_img_name}")
        
        # 将目标图像ID转换为文件名（添加.jpg后缀）
        tgt_img_name = f"{tgt_img_name}.jpg"  # 转换为字符串并添加.jpg
        
        #print(src_img_name,tgt_img_name)
        
        # 加载图像
        src_img_path = os.path.join(self.src_dir, src_img_name)
        tgt_img_path = os.path.join(self.target_dir, tgt_img_name)
        
        src_img = Image.open(src_img_path).convert('RGB')
        tgt_img = Image.open(tgt_img_path).convert('RGB')
        
        # 应用变换
        if self.transform is not None:
            src_img = self.transform(src_img)
            tgt_img = self.transform(tgt_img)
        
        return src_img, tgt_img

# 使用示例
if __name__ == "__main__":
    # 配置路径
    src_dir = "tradesy_clip_select/src"  # 存放源图像的目录
    ref_results_file = "/data/lz/AIP/data/reference_selection_results.txt"  # 参考选择结果文件
    target_dir = "tradesy_clip_select/target"  # 存放目标图像的目录
    
    # 创建数据集实例
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    

    
    dataset = PairedImageDataset(src_dir, ref_results_file, target_dir, transform)
    print("Dataset length:", len(dataset))
    src, tgt = dataset[0]  # 测试第一个样本
    print("Source image shape:", src.shape)
    print("Target image shape:", tgt.shape)