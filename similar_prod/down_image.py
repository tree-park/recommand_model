import requests  # to get image from the web
import shutil  # to save it locally
from torchvision import datasets, transforms
import pandas as pd
import torch
import numpy as np

# 경고 메시지 무시하기
import warnings

warnings.filterwarnings("ignore")

# Set up the image URL and filename
filepath = '../data/bungae_test/images/'


def get_image(image_url):
    filename = image_url[0]

    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url[1], stream=True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with wb ( write binary ) permission.
        with open(filepath + filename + '.jpg', 'wb') as f:
            shutil.copyfileobj(r.raw, f)

        print('Image sucessfully Downloaded: ', filepath + filename)
    else:
        print('Image Couldn\'t be retreived')


class ImgDataset:
    def __init__(self, dataset):
        self.dataset = {}
        for fp, img in zip(dataset.imgs, dataset):
            fn = fp[0].split('/')[-1].replace('.jpg', '')
            self.dataset[fn] = img[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        return self.dataset[id]


def read_image(dirpath):
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

    imgdata = datasets.ImageFolder(dirpath, transform=transform)
    imgdata = ImgDataset(imgdata)
    return imgdata


def read_text(filepath):
    data = pd.read_csv(filepath,
                       compression='gzip',
                       quotechar='"',
                       escapechar='\\',
                       dtype=str,
                       nrows=100
                       )

    data.dropna(subset=['image_url'], inplace=True)
    df = data['content_id'].groupby(data['content_id']).count()

    b = data.sort_values(['content_id', 'ref_term']).groupby('content_id', as_index=False).first()
    b.fillna('', inplace=True)

    # text 구성 <sep> 없어도 괜찮은걸까 확인필요...
    b['texts'] = '<cls> ' + b['name'] + ' <kwd> ' + b['keyword'] + ' <cat> ' + b['category_name']

    train_text = {cid: texts for cid, texts in b[['content_id', 'texts']].values}
    return train_text
