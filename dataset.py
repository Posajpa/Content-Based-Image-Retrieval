import torch
import torchvision
import glob
import os
import pandas as pd
import PIL.Image as Image
import numpy as np
import cv2



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
        # img = crop_face(img)
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

def crop_face(input_image):
    # convert to grayscale of each frames
    input_image_np = np.array(input_image)
    gray = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2GRAY)

    # read the haarcascade to detect the faces in an image
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    # detects faces in the input image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    cropped_face = []
    # loop over all detected faces
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            # To draw a rectangle in a face
            dx = w * 0.1
            dy = h * 0.1
            cv2.rectangle(input_image_np, (x, y), (x + w, y + h), (0, 255, 255), 2)
            face = input_image_np[y:y + h, x:x + w]
            cropped_face.append(face)
    if len(cropped_face) != 0:
        out = Image.fromarray(cropped_face[0])
    else:
        out = input_image
    return out

class competitionSet(torch.utils.data.Dataset):
    def __init__(self, path):
        self.root = path
        self.image_paths = glob.glob(os.path.join(path, "*.*"))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224), antialias=True),
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
        img = crop_face(img)
        img = self.transform(img)
        img_name = os.path.basename(self.image_paths[idx])
        return img, img_name


def get_test_data(config):
    query_set = competitionSet("C:/Users/Posajpa/PycharmProjects/Machine_Learning/test_data/query")
    gallery_set = competitionSet("C:/Users/Posajpa/PycharmProjects/Machine_Learning/test_data/gallery")


    # data loaders
    query_loader = torch.utils.data.DataLoader(query_set, config["batch_size"], shuffle=False)
    gallery_loader = torch.utils.data.DataLoader(gallery_set, config["batch_size"], shuffle=False)

    return query_loader, gallery_loader

def get_query_and_gallery(config):
    query_set = competitionSet("C:/Users/Posajpa/PycharmProjects/Machine_Learning/test_data/query")
    gallery_set = competitionSet("C:/Users/Posajpa/PycharmProjects/Machine_Learning/test_data/gallery")

    return query_set, gallery_set
