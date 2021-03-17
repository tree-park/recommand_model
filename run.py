
import warnings
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models as torch_vmodels
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from down_image import get_image
from preprocessing import TextProcessor, TextImageDataset, to_batch, read_image, read_text
from model import TextImg2Vec


pd.set_option('display.max_columns', 10)
warnings.filterwarnings("ignore")

data = pd.read_csv('data/bungae_test/rec-exam.csv000.gz',
                   compression='gzip',
                   quotechar='"',
                   escapechar='\\',
                   dtype=str,
                   # nrows=3000
                   )
data.dropna(subset=['image_url'], inplace=True)
# 139687586

# download data
# for v in zip(data['content_id'], data['image_url']):
#     # print(v)
#     get_image('../data/bungae_test/images/', v)

img = read_image('data')
text = read_text(data)

text_model, vocab = get_pytorch_kobert_model()
tok_path = get_tokenizer()
tokenizer = SentencepieceTokenizer(tok_path)

tk = TextProcessor(vocab, tokenizer)
for cid in text.keys():
    text[cid] = tk.text_tokenizing(text[cid])

dataset = TextImageDataset(text, img)
data_loader = DataLoader(dataset, batch_size=100, collate_fn=to_batch)

img_model = torch_vmodels.resnet18(pretrained=True)
vec_model = TextImg2Vec(img_model, text_model)
vec_model.to('cpu')
vec_model.eval()

print("Start vectorizing")
results = {}
for inputs in data_loader:
    text_id, img = [x.to('cpu') for x in inputs[1:]]
    output = vec_model(text_id, img)
    new = {cid: vec[0] for cid, vec in zip(inputs[0], output.split(1))}
    results = {**results, **new}

try:
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
except Exception as err:
    print(err)

print("Calculate similarity")
all_ = {}
for cid1, vec in results.items():
    his = []
    for cid2, comp in results.items():
        sim = F.cosine_similarity(vec, comp, 0)
        his.append((cid2, sim))
    all_[cid1] = sorted(his, key=lambda x: x[1], reverse=True)[:10]

print(all_)
