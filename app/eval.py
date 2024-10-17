from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim

from test import *
from utils import *

# Load the .npz dataset
images, labels = decompress_npz('../data/compressed_asl_crop.npz')

# Split the images and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load TorchScript model
model = CustomResnet18(len(np.unique(labels)))
filepath = '../data/checkpoint_epoch_10.pth'
optimizer = optim.Adam(model.parameters(), lr=0.001)

load_checkpoint(filepath, model, optimizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=mean, std=std)
])

batch_size = 100

test_dataset = GestureDataset(X_test, y_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

evaluate_model(model, test_loader, device)
