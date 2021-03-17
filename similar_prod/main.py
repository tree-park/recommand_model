import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import TensorDataset


from similar_prod.down_image import get_image, read_image, read_text
from similar_prod.kobert_test import TextProcessor, TextImageDataset
pd.set_option('display.max_columns', 10)

# download data
# for v in zip(data['content_id'], data['image_url']):
#     print(v)
#     get_image(v)

img = read_image('../data')
text = read_text('../data/bungae_test/rec-exam.csv000.gz')
dataset = TextImageDataset(text, img)
dloader = DataLoader(dataset)











# merge text, image by prod id

