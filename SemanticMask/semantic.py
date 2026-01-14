import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class SemanticSegmentation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        # 使用预训练好的语义分割模型
        self.model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
        self.model.eval()
        self.model.cuda()

    def forward(self, x):
        # (B, 3, 512, 512)
        x=x.cuda()
        outputs = self.model(x)
        imsize = x.shape[2]
        inputs = F.interpolate(input=outputs, size=(imsize, imsize), mode='bilinear', align_corners=True)

        # 获取每个像素的预测类别（这里是二分类问题，背景 vs. 前景）
        pred_batch = torch.argmax(inputs, dim=1)  # (B, 512, 512)
        
        return pred_batch

class SemanticWithoutSegmentation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 更换更适合服装分割的模型（关键修改1）
        self.model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
        self.model.eval()
        self.model.cuda()

    def forward(self, x):
        with torch.no_grad():
          outputs = self.model(x)['out']  # 输出形状 [1, 21, 512, 512]
        return outputs.argmax(1, keepdim=True)  # 保持维度 [1, 1, 512, 512]


if __name__ == '__main__':
    # 预处理修正（关键修改2）
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # 使用ImageNet标准归一化参数
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    # 加载图像并预处理
    input_image = Image.open('./140463.jpg')
    input_tensor = transform(input_image).unsqueeze(0).cuda()
    print(input_tensor.shape)
    
    # 初始化模型
    model = SemanticSegmentation()
    mask = model(input_tensor)
    print(mask.shape)
    # 生成mask逻辑简化（关键修改3）
    #mask = (output_mask == 15).astype(np.uint8)  # COCO person类索引为15
    mask = (mask == 15).float()  # 深蓝色T恤区域设为0，背景设为1
    # 生成被mask的图像（形状 [1, 3, 512, 512]）
    masked_image = input_tensor * (mask)  # mask<0.5的区域保留（即T恤区域）
    print("Masked image shape:", masked_image.shape)  # 应输出 torch.Size([1, 3, 512, 512])
    
   # 转换为numpy并保存
    result = masked_image[0].permute(1, 2, 0).cpu().numpy()
    result = (result * 255).astype(np.uint8)
    cv2.imwrite('masked_result.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))