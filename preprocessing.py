import torch
import torch.nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ImgDataset:
    def __init__(self, dataset):
        self.dataset = {}
        for fp, img in zip(dataset.imgs, dataset):
            fn = fp[0].split('/')[-1].replace('.jpg', '')
            self.dataset[fn] = img[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, cid):
        return self.dataset[cid]


def read_image(dirpath):
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

    imgdata = datasets.ImageFolder(dirpath, transform=transform, )
    imgdata = ImgDataset(imgdata)
    return imgdata


def read_text(dataframe):
    b = dataframe.sort_values(['content_id', 'ref_term']).groupby('content_id',
                                                                  as_index=False).first()
    b.fillna('', inplace=True)

    # TODO tag [SEP]', '[CLS]
    b['texts'] = '[CLS] ' + b['name'] + ' [kwd] ' + b['keyword'] + ' [kwd] ' + b['category_name']

    train_text = {cid: texts for cid, texts in b[['content_id', 'texts']].values}
    return train_text


class TextProcessor:
    def __init__(self, vocab, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = vocab

    def text_tokenizing(self, text):
        idx = self.vocab.to_indices(self.tokenizer(text))
        # attn_id = torch.ones(1, len(idx))) # TODO att 정보 만들기
        return idx


class TextImageDataset(Dataset):
    def __init__(self, textdata, imgdata):
        # assert len(textdata) == len(imgdata)
        self.textdata = textdata
        self.imgdata = imgdata
        self.data = []
        for idx, cid in enumerate(imgdata.dataset.keys()):
            self.data.append(
                [cid, torch.tensor(self.textdata[cid]), torch.tensor(self.imgdata[cid])])

    def __len__(self):
        return len(self.textdata)

    def __getitem__(self, idx):
        return self.data[idx]


def to_batch(batch):
    cid, text, img = zip(*batch)
    pad_ko = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)
    return cid, pad_ko, torch.stack(img)
