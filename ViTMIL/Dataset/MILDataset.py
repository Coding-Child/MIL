import os
import random
from PIL import Image
import pandas as pd
import json

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

from util.mask import *
from util.pseudo_label import get_pseudo_label

Image.MAX_IMAGE_PIXELS = None


def collate_fn(batch):
    # 배치에서 각 항목을 분리합니다.
    images, labels, pseudo_labels = zip(*batch)

    # 모든 이미지에서 최대 패치 수를 찾습니다.
    max_patches = max([image.shape[0] for image in images])

    # 각 이미지를 최대 패치 수에 맞게 패딩합니다.
    padded_images = [pad_image_to_max_patches(image, max_patches) for image in images]
    padded_pseudo_labels = [pad_pseudo_labels(p_label, max_patches) for p_label in pseudo_labels]

    # 패딩된 이미지들을 스택하여 배치 텐서를 만듭니다.
    images_tensor = torch.stack(padded_images)

    # 패딩된 pseudo_labels를 스택하여 배치 텐서를 만듭니다.
    pseudo_labels_tensor = torch.stack(padded_pseudo_labels)

    # 레이블을 텐서로 변환합니다.
    labels_tensor = torch.tensor(labels)

    return images_tensor, labels_tensor, pseudo_labels_tensor


def pad_image_to_max_patches(image, max_patches):
    # 이미지의 패치 수가 최대 패치 수보다 적으면 패딩을 추가합니다.
    # 각 패치는 (C, H, W)의 형태를 가지고 있으므로, 패딩은 (0, 0, 0, 0, 0, 0, P, 0)의 형태로 추가됩니다.
    # 여기서 P는 패딩할 패치의 수입니다.
    padding_patches = max_patches - image.shape[0]
    if padding_patches > 0:
        # 패치의 크기를 가져옵니다. (C, H, W)
        patch_size = image.shape[1:]
        # 패딩할 패치를 생성합니다. 여기서는 0으로 채워진 패치를 사용합니다.
        padding = torch.zeros((padding_patches, *patch_size), dtype=image.dtype)
        # 원본 이미지와 패딩을 결합합니다.
        image = torch.cat([image, padding], dim=0)
    return image


def pad_pseudo_labels(pseudo_labels, max_patches):
    if pseudo_labels.dim() == 0:
        pseudo_labels = pseudo_labels.unsqueeze(0)

    # 패딩할 레이블의 수를 계산합니다.
    padding_labels = max_patches - len(pseudo_labels)
    if padding_labels > 0:
        # 0으로 채워진 레이블을 생성합니다.
        padding = torch.zeros(padding_labels, dtype=pseudo_labels.dtype)
        # 원본 레이블과 패딩을 결합합니다.
        pseudo_labels = torch.cat([pseudo_labels, padding], dim=0)
    return pseudo_labels


def generate_patches_json(path, patch_size, save_path, static, model, n_clusters):
    """
    Generate patches information for a given image and save it to a JSON file.

    :param path: path of the image data
    :param patch_size: The size of the patch (will be the same for height and width)
    :param save_path: path of save the json file
    :param static: static pseudo label
    :model: Deep Neural Network Model for extract the embedding
    :n_clusters: Number of Cluster
    """
    img_label = path.split('\\')[1]
    img_name = path.split('\\')[-1].split('.')[0]

    img = Image.open(path)
    np_img = np.array(img)

    W, H = img.size
    patches_info = dict()
    patches_list = list()

    # Calculate the number of patches along height and width
    num_patches_height = (H + patch_size - 1) // patch_size  # Ceiling division
    num_patches_width = (W + patch_size - 1) // patch_size  # Ceiling division
    patch_number = 0

    for i in range(num_patches_height):
        for j in range(num_patches_width):
            y = i * patch_size
            x = j * patch_size
            patch_key = f"patch_{patch_number}"
            patch = np_img[y:y + patch_size, x:x + patch_size]

            # Calculate the tissue mask and the percentage of tissue in the patch
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                tissue_mask_patch = tissue_mask(patch)
                tissue_percentage = tissue_percent(tissue_mask_patch)

                # If the tissue percentage is greater than 20%, add to the JSON info
                if tissue_percentage >= 20:
                    patches_info[patch_key] = {
                        "location": (y, x),
                        "size": (patch_size, patch_size)
                    }
                    patches_list.append(Image.fromarray(patch))
            patch_number += 1

    if static:
        if patches_list and int(img_label) == 1:
            pseudo_labels = get_pseudo_label(model, patches_list, n_clusters)
            for i, patch_key in enumerate(patches_info.keys()):
                patches_info[patch_key]['pseudo_label'] = int(pseudo_labels[i])
        elif patches_list and int(img_label) == 0:
            for i, patch_key in enumerate(patches_info.keys()):
                patches_info[patch_key]['pseudo_label'] = 0
    else:
        pseudo_labels = get_pseudo_label(model, patches_list, n_clusters)
        for i, patch_key in enumerate(patches_info.keys()):
            patches_info[patch_key]['pseudo_label'] = int(pseudo_labels[i])

    try:
        json_filename = f'{save_path}/{img_label}/{img_name}.json'
        os.makedirs(os.path.dirname(json_filename), exist_ok=True)  # Ensure the directory exists
        with open(json_filename, 'w') as json_file:
            json.dump(patches_info, json_file, indent=4)
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


def extract_patches(json_path, img_path, num_samples):
    pseudo_labels = list()

    # JSON 데이터 로드
    with open(json_path, 'r') as file:
        patches_info = json.load(file)

    patch_names_label_1 = [name for name, data in patches_info.items() if data['pseudo_label'] == 1]
    patch_names_label_0 = [name for name, data in patches_info.items() if data['pseudo_label'] == 0]
    all_patch_names = patch_names_label_0 + patch_names_label_1

    # 사용 가능한 패치 수가 요청된 샘플 수보다 적은 경우, 모든 패치를 사용합니다.
    if len(patch_names_label_1) == 0:
        if len(patch_names_label_0) >= num_samples:
            sampled_patch_names = random.sample(patch_names_label_0, num_samples)
        else:
            sampled_patch_names = patch_names_label_0
    else:
        num_label_1 = int(num_samples * 0.7)  # pseudo_label이 1인 패치의 수
        num_label_1 = min(num_label_1, len(patch_names_label_1))  # 사용 가능한 최대 수

        sampled_patch_names_label_1 = random.sample(patch_names_label_1, num_label_1)
        remaining_samples = num_samples - num_label_1
        remaining_samples = min(remaining_samples, len(patch_names_label_0))

        sampled_patch_names_label_0 = random.sample(patch_names_label_0, remaining_samples) \
            if len(patch_names_label_0) > 0 else random.sample(all_patch_names, remaining_samples)

        # sampled_patch_names_label_0에서 샘플링된 항목 제외
        remaining_patches = list(set(all_patch_names) - set(sampled_patch_names_label_0))
        additional_samples = remaining_samples - len(sampled_patch_names_label_0)
        additional_samples = min(additional_samples, len(remaining_patches))

        if additional_samples > 0:
            additional_sampled_patches = random.sample(remaining_patches, additional_samples)
            sampled_patch_names = sampled_patch_names_label_1 + sampled_patch_names_label_0 + additional_sampled_patches
        else:
            sampled_patch_names = sampled_patch_names_label_1 + sampled_patch_names_label_0

    # 이미지 파일을 엽니다.
    with Image.open(img_path) as img:
        # 추출된 패치를 저장할 리스트
        patches_list = []

        for patch_name in sampled_patch_names:
            patch_data = patches_info[patch_name]
            # JSON 파일에서 제공된 위치와 크기 정보를 사용하여 영역을 계산합니다.
            x, y = patch_data['location']
            width, height = patch_data['size']
            pseudo_label = patch_data['pseudo_label']
            # 실제 시작 위치를 계산합니다.
            start_x = x
            start_y = y
            # 이미지에서 영역을 추출합니다.
            patch = img.crop((start_x, start_y, start_x + width, start_y + height))
            # 추출된 영역을 리스트에 추가합니다.
            patches_list.append(patch)
            pseudo_labels.append(pseudo_label)

        if not patches_list:
            raise ValueError(f"No patches found path {img_path}")

        # 추출된 패치 리스트를 반환합니다.
        return patches_list, pseudo_label


def data_split(csv_file):
    data_info = pd.read_csv(csv_file)
    train_df, val_test_df = train_test_split(data_info, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)

    return train_df, val_df, test_df


class PathologyDataset(Dataset):
    def __init__(self, df, num_samples, transform=None, test=False):
        self.data_info = df
        self.test = test
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # 해당 인덱스의 이미지 경로와 JSON 경로, 레이블을 가져옴
        img_path = self.data_info.iloc[idx]['img_path']
        json_path = self.data_info.iloc[idx]['json_path']
        label = self.data_info.iloc[idx]['label']

        # 이미지에서 패치 추출 (위에서 정의한 함수 사용)
        patches, pseudo_label = extract_patches(json_path, img_path, self.num_samples)

        # transform이 있으면 각 패치에 적용
        if self.transform:
            patches_tensor = torch.stack([self.transform(patch) for patch in patches])
        else:
            patches_tensor = torch.stack([T.ToTensor()(patch) for patch in patches])

        # 레이블을 텐서로 변환
        label = torch.tensor(label, dtype=torch.long)
        pseudo_label = torch.tensor(pseudo_label, dtype=torch.long)

        # 패치와 레이블 반환
        return patches_tensor, label, pseudo_label
