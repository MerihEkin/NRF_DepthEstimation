import h5py
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Implement bilateral filtering, Spatial Distance (σ_s): The value is set to 3. 
# Intensity Difference (σ_r): The value is set to 30.
# Implement window size and location window size 150 for NYU dataset
# Implement DataLoader Class and iterable object in Pytorch
# 

class NYUV2Dataset:
    def __init__(self, data_file : str, test_size = 0.2, window_size = 150, transforms = None) -> None:
        self.data_file = data_file
        self.test_size = test_size
        self.window_size = window_size

        self.increment = window_size // 2

        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data_to_arrays()

        self.k, self.n, self.m = self.X_train[0].shape

        i = np.arange(window_size // 2, self.m  - window_size // 2, 1)
        j = np.arange(window_size // 2, self.n  - window_size // 2, 1)
        x, y = np.meshgrid(i, j)
        self.xs = x.flatten()
        self.ys = y.flatten()

        self.transforms = transforms


    def __len__(self):
        return self.xs.size * (len(self.X_train) - 1)


    def __getitem__(self, idx):
        image_index = idx // self.xs.size
        pixel_index = idx % self.xs.size

        img = self.X_train[image_index]
        i, j = self.xs[pixel_index], self.ys[pixel_index]

        current_window = img[:, j - self.window_size//2:j + self.window_size//2, i - self.window_size//2:i + self.window_size//2]

        if self.transforms:
            current_window = self.transforms(current_window)

        return current_window, self.y_train[image_index][j, i]


    def load_data_to_arrays(self):
        data = h5py.File(self.data_file, 'r')
        
        # Ensure that the 'depths' and 'images' datasets have the same length
        assert len(data['depths']) == len(data['images'])
        
        # Get the number of images (and depths, since their count is the same)
        num_image = len(data['images'])
        
        images = []  
        depths = []

        for i in range(num_image):
            image = data['images'][i].astype(np.float32) / 255.0
            depth = data['depths'][i].astype(np.float32)
            images.append(image) # add maybe .T
            depths.append(depth) # add maybe .T

            # image_PIL = Image.fromarray(X_train[i].T)
            # depth_PIL = Image.fromarray(np.uint8(y_train[i].T/np.max(y_train[i]) * 255), 'L')
            # image_arr = np.array(image_PIL)
            # depth_arr = np.array(depth_PIL)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(images, depths, test_size=self.test_size)

        return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    
    dataset = NYUV2Dataset('data/nyu_depth_v2_labeled.mat')
    # Create DataLoader
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    train_features, train_labels = next(iter(train_dataloader))
