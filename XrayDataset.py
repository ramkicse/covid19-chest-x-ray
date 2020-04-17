from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image


class XrayDataset(Dataset):
    
    def __init__(self, base, folder,  csv_path, transform=None):
        
        self.base = base
        self.folder = folder
        self.df = pd.read_csv(os.path.join(os.getcwd(), base, csv_path))
        self.transform = transform
        
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        filepath = self.df.iloc[idx].filename
        filepath = os.path.join(os.getcwd(), self.base, self.folder,  filepath)
        class_id = self.df.iloc[idx].label
        #print(filepath)
        image =  Image.open(filepath).convert('RGB')

        #image = cv2.resize(image, (224, 224))

        if self.transform:
            image = self.transform(image)
            #print(image.shape)
        #print(image.shape)
        return image, class_id
