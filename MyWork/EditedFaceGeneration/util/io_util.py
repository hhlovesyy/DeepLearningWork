import numpy as np
import imageio
import torch,os

def save_tensor2img(path,img):
    root = os.path.dirname(path)
    os.makedirs(root,exist_ok=True)
    if isinstance(img,torch.Tensor):
        img_ = img.detach().clone()
    elif isinstance(img,np.ndarray):
        img_ = torch.tensor(img)

    if img_.dtype == torch.uint8:
        img_ = img_ /255.

    if img_.dim() == 4:
        img_ = img_[0]
    if img_.shape[0] == 3 or img_.shape[0] == 1:
        img_ = img_.permute(1,2,0)
    # imageio.imwrite(path,img_.clip(0,1).detach().cpu().numpy())
    # imageio.imwrite(path,(img_.clip(0,1).detach().cpu().numpy()*255.).astype(np.uint8))
    import cv2
    # 假设 img_ 是一个包含 RGB 数据的 Tensor
    img_bgr = img_.clip(0, 1).detach().cpu().numpy() * 255.
    img_bgr = img_bgr.astype(np.uint8)

    # 将 RGB 转换为 BGR
    img_bgr = img_bgr[..., ::-1]  # 对最后一个维度进行反转

    # 使用 OpenCV 保存图像
    cv2.imwrite(path, img_bgr)  # 这里有个坑，因为OpenCV似乎是以BGR格式存储RGB图像的
    

def save_dp2img(path,img):
    img_ = img.detach().clone()
    if img_.dim() == 4:
        img_ = img_[0]
    if img_.shape[0] == 3:
        img_ = img_.permute(1,2,0)
    # imageio.imwrite(path,img_.clip(0,1).detach().cpu().numpy())
    imageio.imwrite(path,((img_*0.5+1).clip(0,1).detach().cpu().numpy()*255.).astype(np.uint8))
