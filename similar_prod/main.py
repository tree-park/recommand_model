import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torch_vmodels
from torch.utils.data import TensorDataset

from similar_prod.down_image import get_image, read_image, read_text
from similar_prod.kobert_test import TextProcessor, TextImageDataset, to_batch
from gluonnlp.data import SentencepieceTokenizer

pd.set_option('display.max_columns', 10)

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from similar_prod.kobert_test import TextProcessor
from similar_prod.model import TextImg2Vec

data = pd.read_csv('../data/bungae_test/rec-exam.csv000.gz',
                   compression='gzip',
                   quotechar='"',
                   escapechar='\\',
                   dtype=str,
                   # nrows=100
                   )
data.dropna(subset=['image_url'], inplace=True)

# download data
for v in zip(data['content_id'], data['image_url']):
    print(v)
    get_image(v)

img = read_image('../data')
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

results = {}
for inputs in data_loader:
    # print(inputs)
    text_id, img = [x.to('cpu') for x in inputs[1:]]
    output = vec_model(text_id, img)
    # results[inputs[0]] = output.split(1)
    new = {cid: vec[0] for cid, vec in zip(inputs[0], output.split(1))}
    results = {**results, **new}

# result = torch.stack(results).squeeze(1)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

all_ = {}
for cid1, vec in results.items():
    his = []
    for cid2, comp in results.items():
        sim = F.cosine_similarity(vec, comp, 0)
        his.append((cid2, sim))
    all_[cid1] = sorted(his, key=lambda x: x[1], reverse=True)[:10]

print(all_)

# idx = vocab.to_indices(sp('[CLS] 한국어 모델을 공유합니다. [SEP]'))
# sequence_output, pooled_output = model(torch.tensor([idx]), torch.ones(1, len(idx)))


# merge text, image by prod id
