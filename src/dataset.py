from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import re

class FocusStackingDataset(Dataset):
    """
    A PyTorch Dataset class to load focus stacking images and their corresponding labels.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples(root_dir)

    def _load_samples(self, root_dir):
        """
        Helper method to load image paths and sort them accordingly.
        """
        files = os.listdir(root_dir)
        samples = {}
        # Regex to extract batch_nr and rolling_nr
        pattern = re.compile(r'(\d+)_(\d+)_')
        
        for filename in files:
            match = pattern.search(filename)
            if match:
                batch_nr, rolling_nr = match.groups()
                key = (int(batch_nr), int(rolling_nr))
                if key not in samples:
                    samples[key] = {'images': [], 'label': None}

                if filename.endswith('GT_pose_0_thermal.png'):
                    samples[key]['label'] = os.path.join(root_dir, filename)
                elif filename.endswith('.png') and 'integral' in filename:
                    samples[key]['images'].append(os.path.join(root_dir, filename))

        # Sort the images by the filename which implicitly sorts by the focal plane due to naming convention
        for key in samples:
            samples[key]['images'].sort()

        # Convert dict to list of tuples and filter out incomplete samples
        return [(k, v['images'], v['label']) for k, v in samples.items() if len(v['images']) == 4 and v['label']]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        batch_nr, image_paths, label_path = self.samples[idx]
        
        # Load images and label
        images = [Image.open(path) for path in image_paths]
        label = Image.open(label_path)
        
        # Apply transformations
        if self.transform is not None:
            images = [self.transform(image) for image in images]
            label = self.transform(label)

        return images, label

# Example usage:
# Set the directory where the images are stored
root_dir = 'path/to/image/folder'

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # Add any other transformations you need
])

# Create the dataset
dataset = FocusStackingDataset(root_dir=root_dir, transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate through the data
for images, label in dataloader:
    # images is a list of 4 tensors
    # label is a tensor
    # Do something with the images and label...
    pass  # Replace with your code to handle the batch