import torch
import torch.nn
from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer

from torch.utils.data import Dataset, DataLoader

# input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
# input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
# token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
model, vocab = get_pytorch_kobert_model()
tok_path = get_tokenizer()

sp = SentencepieceTokenizer(tok_path)
# idx = vocab.to_indices(sp('[CLS] 한국어 모델을 공유합니다. [SEP]'))
# sequence_output, pooled_output = model(torch.tensor([idx]), torch.ones(1, len(idx)))
Model, Vocab = get_pytorch_kobert_model()


class TextProcessor:
    def __init__(self, vocab, tokenizer):
        self.tokenizer = SentencepieceTokenizer(tokenizer)
        self.vocab = vocab

    def text_tokenizing(self, text):
        # 텍스트 들어오면 text tokenizing
        # idx = vocab.to_indices(sp('[CLS] 한국어 모델을 공유합니다. [SEP]'))
        idx = vocab.to_indices(sp(text))
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
            self.data.append([self.textdata[cid], self.imgdata[cid]])

    def __len__(self):
        return len(self.textdata)

    def __getitem__(self, idx):
        return self.data[idx]


def make_pad(samples):
    print(samples)
    return

# 0. unsupervised learning
# 두 pretrained 모델 합쳐서 비슷한 임베딩 id 찾기
# 텍스트 만으로 비슷한거 찾기

# 1. 오토인코더 적용 하기

# 2. 학습데이터로 카테고리 id 만들어서 학습해보기

# 세 결과에 대해 주피터로 돌려보고 제출하기 끝
