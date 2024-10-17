from test import *
from compressor import *
from model import *

# Load the .npz dataset
images, labels = decompress_npz('../data/compressed_asl_crop.npz')

# Split the images and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load the model weights
model = CustomResnet18(len(np.unique(labels)))
state_dict = torch.load('../data/gesture_model_weights_epoch_10.pth', weights_only=True)
model.load_state_dict(state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=mean, std=std)
])

test_dataset = GestureDataset(X_test, y_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

evaluate_model(model, test_loader, device)
