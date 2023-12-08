import glob

from tqdm import tqdm
from Dataset.MILDataset import generate_patches_json

from model.ResNet import ResNet50

paths = glob.glob('cancer_data\\*\\**.jpg')
paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

model = ResNet50('simclr_pretrained_model_ckpt/checkpoint_0200_Adam.pt').cuda()


with tqdm(total=len(paths)) as pbar:
    for path in paths:
        generate_patches_json(path, 224, save_path='cancer_data', model=model, n_clusters=2, static=True)

        pbar.update(1)
