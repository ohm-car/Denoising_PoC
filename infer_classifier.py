import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision import models
from tqdm import tqdm

import torchxrayvision as xrv

from datasets.covid_dataset import get_covid_loaders
from datasets.nih_dataset import get_nih_loaders


def build_model(dataset_name: str, class_count: int) -> torch.nn.Module:
    if dataset_name == 'covid':
        model = xrv.models.DenseNet(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, class_count)
        return model

    if dataset_name == 'nih':
        model = models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, class_count)
        return model

    raise ValueError("Invalid dataset choice. Choose either 'nih' or 'covid'.")


def infer_covid(model, loader, device):
    all_probs = []
    all_labels = []

    model.eval()
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc='Infer', leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.asarray(np.concatenate(all_labels))
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    y_true_one_hot = np.eye(y_prob.shape[1])[y_true]
    auc = roc_auc_score(y_true_one_hot, y_prob, multi_class='ovr', average='macro')

    return accuracy, auc


def infer_nih(model, loader, device):
    all_probs = []
    all_labels = []

    model.eval()
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc='Infer', leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_labels)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_true.reshape(-1), y_pred.reshape(-1))

    aucs = []
    for class_index in range(y_true.shape[1]):
        try:
            aucs.append(roc_auc_score(y_true[:, class_index], y_prob[:, class_index]))
        except ValueError:
            aucs.append(np.nan)

    auc = float(np.nanmean(aucs))
    return accuracy, auc


def main():
    parser = argparse.ArgumentParser(description='Run inference for the classifier and report Accuracy and ROC/AUC')
    parser.add_argument('--dataset', type=str, choices=['nih', 'covid'], required=True,
                        help='Dataset the model was trained on: nih or covid')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model .pth file')
    parser.add_argument('-b', '--batch_size', type=int, default=None,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    parser.add_argument('-p', '--p_count', type=int, choices=[0, 12000, 1200, 200], default=0,
                        help='Photon intensity value')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='test',
                        help='Dataset split to run inference on')
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.dataset == 'covid':
        loaders, class_names = get_covid_loaders(
            batch_size=args.batch_size if args.batch_size is not None else 40,
            resize_to=224,
            intensity=args.p_count,
        )
        loader = loaders[args.split]
        class_count = len(class_names)
        model = build_model('covid', class_count)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)

        accuracy, auc = infer_covid(model, loader, device)

        print(f'Dataset: covid')
        print(f'Split: {args.split}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'ROC/AUC (macro OVR): {auc:.4f}')
        return

    if args.dataset == 'nih':
        loaders, pathologies = get_nih_loaders(
            csv_path='./Data/NIH_Chest_XRay/Data_Entry_2017.csv',
            img_dir='./Data/NIH_Chest_XRay/images',
            batch_size=args.batch_size if args.batch_size is not None else 12,
            resize_to=512,
            intensity=args.p_count,
        )
        loader = loaders[args.split]
        class_count = len(pathologies)
        model = build_model('nih', class_count)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)

        accuracy, auc = infer_nih(model, loader, device)

        print(f'Dataset: nih')
        print(f'Split: {args.split}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'ROC/AUC (macro per-label): {auc:.4f}')
        return

    raise ValueError("Invalid dataset choice. Choose either 'nih' or 'covid'.")


if __name__ == '__main__':
    main()