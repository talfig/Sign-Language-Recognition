# test/test_model.py

from test import *

# Load the .npz dataset
X, y = decompress_npz('../data/compressed_asl_crop_v2.npz')

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('../data/asl_crop_v2_mobilenet_weights_epoch_10.pth', device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ASLDataset(X, y, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

evaluate_model(model, test_loader, device)
