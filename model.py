import torch
import torch.nn as nn


class TextImg2Vec(nn.Module):
    def __init__(self, img2vec, text2vec):
        super(TextImg2Vec, self).__init__()
        self.img2vec = img2vec
        self.text2vec = text2vec

    def forward(self, text_id, img):

        img_vec = self.img2vec(img)
        text_states, _ = self.text2vec(text_id)
        cls_vec = text_states[:, 0]
        return torch.cat([img_vec, cls_vec], dim=1)


class Img2Vec(nn.Module):
    def __init__(self, model):
        super(Img2Vec, self).__init__()
        self.model = model

        self.layer = self.model._modules.get('avgpool')
        # Todo nomalized ??

    def forward(self, img_tensor):
        return self.model(img_tensor)


class Text2Vec(nn.Module):
    def __init__(self, model):
        super(Text2Vec, self).__init__()
        self.model = model

    def forward(self, text_tensor):
        return self.model(text_tensor)
