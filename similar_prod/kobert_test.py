import torch
import torch.nn

from torch.utils.data import Dataset, DataLoader



class TextProcessor:
    def __init__(self, vocab, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = vocab

    def text_tokenizing(self, text):
        idx = self.vocab.to_indices(self.tokenizer(text))
        # attn_id = torch.ones(1, len(idx))) # TODO att 정보 만드는 법 참조
        return idx


# 데이터세트 정의 해서 텍스트, 이미지 데이터 셋 합치기
class TextImageDataset(Dataset):
    def __init__(self, textdata, imgdata):
        assert len(textdata) == len(imgdata)
        self.textdata = textdata
        self.imgdata = imgdata
        self.data = []
        for idx, cid in enumerate(textdata.keys()):
            self.data.append([cid, torch.tensor(self.textdata[cid]), torch.tensor(self.imgdata[cid])])

    def __len__(self):
        return len(self.textdata)

    def __getitem__(self, idx):
        return self.data[idx]


def to_batch(batch):
    cid, text, img = zip(*batch)
    pad_ko = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)
    return cid, pad_ko, torch.stack(img)




# 0. unsupervised learning
# 두 pretrained 모델 합쳐서 비슷한 임베딩 id 찾기
# 텍스트 만으로 비슷한거 찾기

# 1. 오토인코더 적용 하기

# 2. 학습데이터로 카테고리 id 만들어서 학습해보기

# 세 결과에 대해 주피터로 돌려보고 제출하기 끝
