import torch
import torchvision
import glob
import os
import pandas as pd
import PIL.Image as Image

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        df = pd.read_csv(csv_file, delimiter=' ')
        df = df.sort_values(by=['name'])
        self.names = df["name"]
        self.labels = df["label"]
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.*")))
        # self.image_paths = os.listdir(os.path.join(root_dir))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224), antialias=True),
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            # torchvision.transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
            # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.labels)

    def get_unique_labels(self):
        s = set()
        for i in self.labels:
            s.add(i)
        return len(s)

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        name = self.names[idx]
        img = Image.open(self.image_paths[idx]).convert("RGB")
        # img = crop_face(img, name)
        img = self.transform(img)  # preprocessed image
        return img, name, label

def get_train_data(config):
    if not config["val_split_req"]:
        train_set = FaceDataset(csv_file=config["train_data_labels"], root_dir=config["train_data"])
        val_set = FaceDataset(csv_file=config["val_data_labels"], root_dir=config["val_data"])
        out_layer = train_set.get_unique_labels()
    else:
        whole_data = FaceDataset(csv_file=config["train_data_labels"], root_dir=config["train_data"])
        ll = len(whole_data)
        train_split = round(ll * 0.8)
        val_split = round(ll - ll * 0.8)
        train_set, val_set = torch.utils.data.random_split(whole_data, [train_split, val_split])
        out_layer = whole_data.get_unique_labels()

    # Create the dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, config["batch_size"], shuffle=False)

    return train_loader, val_loader, out_layer

class competitionSet(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.root = path
        self.image_paths = glob.glob(self.root + "*.*")
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            # torchvision.transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # open the image
        img = Image.open(self.image_paths[idx]).convert("RGB")  # convert rgb images to grayscale
        img = self.transform(img)

        # get the image name
        # img_name = os.path.basename(self.image_paths[idx]).split(".")[0]
        img_name = os.path.basename(self.image_paths[idx])
        return img, img_name

def get_test_data(config):
    query_set = competitionSet(config["query_data"])
    gallery_set = competitionSet(config["gallery_data"])

    # data loaders
    query_loader = torch.utils.data.DataLoader(query_set, config["batch_size"], shuffle=False)
    gallery_loader = torch.utils.data.DataLoader(gallery_set, config["batch_size"], shuffle=False)

    return query_loader, gallery_loader