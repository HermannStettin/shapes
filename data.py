from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_train_val_test_data(data_params, batch_size, device):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ColorJitter(brightness = 0.3, contrast = 0.3),
        transforms.RandomAffine(degrees = 30, shear = 15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    train_path = data_params["data_path"] / "train"
    test_path = data_params["data_path"] / "test"
    val_path = data_params["data_path"] / "val"

    train_data = datasets.ImageFolder(root = train_path, transform = train_transforms)
    val_data = datasets.ImageFolder(root = val_path, transform = val_transforms)
    test_data = datasets.ImageFolder(root = test_path, transform = test_transforms)

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)
    val_loader = DataLoader(dataset = val_data, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader, val_loader

