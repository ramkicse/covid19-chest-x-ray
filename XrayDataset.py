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
        #image = self.remove_text(image)
        #image = cv2.resize(image, (224, 224))

        if self.transform:
            image = self.transform(image)
            #print(image.shape)
        #print(image.shape)
        return image, class_id
    
    def remove_text(img):
        '''
        Attempts to remove textual artifacts from X-ray images. For example, many images indicate the right side of the
        body with a white 'R'. Works only for very bright text.
        :param img: Numpy array of image
        :return: Array of image with (ideally) any characters removed and inpainted
        '''
        mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)[1][:, :, 0].astype(np.uint8)
        img = img.astype(np.uint8)
        result = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS).astype(np.float32)
        return result
