import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os
import torch

TRAIN_PATH = 'dataset/Train'
TEST_PATH = 'dataset/Test'


# predefine transform
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),         # ResNet expects ~224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means & stds
                std =[0.229, 0.224, 0.225]),
    ])

# load data from files
def load_data(PATH):
    filenames, fruit, fresh = [], [], []
    
    for file in sorted(os.listdir(PATH)):
        if file=='.DS_Store':
            continue
        for img in os.listdir(os.path.join(PATH, file)):
            if not (img.endswith(('.png','.jpeg','.jpg'))):
                continue
            fresh.append(0 if file[0] == 'f' else 1)
            fruit.append(file[5:] if file[0] == 'f' else file[6: ])
            filenames.append(os.path.join(PATH, file, img))
            
    df = pd.DataFrame({
        'filename' : filenames,
        'fruit' : fruit,
        'fresh' : fresh
    })
    
    le = LabelEncoder()
    df['fruit_label'] = le.fit_transform(df['fruit'])
    return df

class FruitDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): Must have columns:
                - 'filename' (str): path to image
                - 'fruit_label' (int): numeric fruit ID
                - 'fresh' (int): 0 (fresh) or 1 (rotten)
            transform (callable): optional torchvision transform
        """
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filename']
        fruit_label = row['fruit_label']
        fresh_label = row['fresh']  # 0 = fresh, 1 = rotten

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms (e.g. resizing, normalization)
        if self.transform:
            image = self.transform(image)

        # Return
        return image, torch.tensor(fruit_label, dtype=torch.long), torch.tensor(fresh_label, dtype=torch.long)


df_test = load_data(TEST_PATH)
df_train = load_data(TRAIN_PATH)

train_dataset = FruitDataset(df_train, transform=resnet_transform)
test_dataset  = FruitDataset(df_test,  transform=resnet_transform)

print(train_dataset.df.head())


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)