
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class Mydataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.image_paths = self.get_image_paths("Normal")
    def __getitem__(self,idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.root_dir,img_name)
        image = Image.open(img_path).convert("RGB")
        image = image.resize((128,128))
        img = transforms.ToTensor()(image)
        # assert img.shape == [3, 512, 512]
        return img
    def __len__(self):
        return len(self.image_paths)
    def get_image_paths(self,class_name):
        image_paths = []
        class_path = os.path.join(self.root_dir,class_name)
        for img_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_name,img_name))
        return image_paths
    

if __name__ == '__main__':
    dataset = Mydataset("TB_Chest_Radiography_Database")
    print(dataset[0].shape)