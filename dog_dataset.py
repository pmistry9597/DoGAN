from torch.utils.data import Dataset
from PIL import Image
import os

class DogData(Dataset):
    def __init__(self, transform=None):
        self.path = 'dogs_cats_sample_1000/train/dogs'
        self.transform = transform

        self.files = [ f for f in os.listdir(self.path) if os.path.isfile(self.path + '/' + f) and f[-3:] == 'jpg' ]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        out = Image.open(self.path + '/' + self.files[idx])
        #out = np.array(img)
        #out = tensor([[npimg], 1])
        if (self.transform):
            out = self.transform(out)
        out = [out, 0]
        return out
