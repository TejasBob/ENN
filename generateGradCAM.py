import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn


from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from PIL import Image
from torchvision import transforms
import torch
import cv2
import numpy as np
import os
os.makedirs("images_train")


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

torch.set_num_threads(4)



def apply_gradcam(img, img_nonnormalized, image_idx):
    target_layers = [model.network.layer3[-1]]
    img = img.permute(1, 2, 0).numpy()
   # img = (img - np.min(img)) / (np.max(img) - np.min(img))
    input_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).float()


    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = None

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True)

    grayscale_cam = grayscale_cam[0, :]
    print(np.min(grayscale_cam), np.max(grayscale_cam))
    rgb_img = img_nonnormalized.permute(1, 2, 0).numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    rgb_img = np.uint8(rgb_img * 255)
    # img1 = F.to_pil_image(img_nonnormalized)
    # img2 = F.to_pil_image(visualization)
    # return [img1, img2]
    print(type(rgb_img), type(visualization))
    return [rgb_img, visualization]


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ResNet
        self.network = models.resnet18(pretrained=True)
        self.network.fc = torch.nn.Linear(512, num_classes)
        self.output = nn.Softplus()

    def forward(self, x):
        logits = self.network(x)
        logits = self.output(logits)
        return logits


best_model_path = "model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNet18(num_classes=10)
model = model.to(device)
# model.load_state_dict(torch.load(best_model_path))
model.eval()
print(model.network)

dataset_mean = [0.4914, 0.4822, 0.4465]
dataset_std = [0.24, 0.24, 0.26]

dataset_normalized = torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize(mean=dataset_mean, std=dataset_std)]))

dataset_non_normalized = torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor()]))


top10_images_normalised = torch.utils.data.Subset(dataset_normalized, list(np.arange(1000)))
top10_images_non_normalized = torch.utils.data.Subset(dataset_non_normalized, list(np.arange(1000)))

img_non_normalized_list, visualization_list = [], []
for i in range(len(top10_images_normalised)):
    img, viz = apply_gradcam(top10_images_normalised[i][0], top10_images_non_normalized[i][0], i)
    img_non_normalized_list.append(img)
    visualization_list.append(viz)
    cv2.imwrite(f"./images_train/img_{i}.png", img[:,:,[2,1,0]])    
    cv2.imwrite(f"./images_train/viz_{i}.png", viz[:,:,[2,1,0]])
print("Done")



