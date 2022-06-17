import torch.nn as nn
import sys
sys.path.append('/public/home/qianych/code/SVIP-Sequence-VerIfication-for-Procedures-in-Videos')

from utils.builder import Builder


class CAT(nn.Module):
    def __init__(self,
                 num_class=20,
                 num_clip=16,
                 dim_embedding=128,
                 pretrain=None,
                 dropout=0,
                 use_TE=False,
                 use_SeqAlign=False,
                 freeze_backbone=False):
        super(CAT, self).__init__()

        self.num_clip = num_clip
        self.use_TE = use_TE
        self.use_SeqAlign = use_SeqAlign
        self.freeze_backbone = freeze_backbone

        module_builder = Builder(num_clip, pretrain, use_TE, dim_embedding)

        self.backbone = module_builder.build_backbone()
        if use_SeqAlign:
            self.seq_features_extractor = module_builder.build_seq_features_extractor()
        if use_TE:
            self.bottleneck = nn.Conv2d(2048, 128, 3, 1, 1)
            self.TE = module_builder.build_transformer_encoder()
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_head = module_builder.build_embed_head()
        self.dropout = nn.Dropout(dropout)
        self.cls_fc = nn.Linear(dim_embedding, num_class)




    def train(self, mode=True):
        """
        Override the default train() to freeze the backbone
        :return:
        """
        super(CAT, self).train(mode)

        if self.freeze_backbone:
            print('Freezeing backbone.')
            for param in self.backbone.parameters():
                param.requires_grad = False


    def forward(self, x, embed=False):


        x = self.backbone(x)  # [bs * num_clip, 2048, 6, 10]

        seq_features = None
        if self.use_SeqAlign:
            seq_features = self.seq_features_extractor(x)

        if self.use_TE:
            x = self.bottleneck(x)
            _, c, h, w = x.size()
            x = x.reshape(-1, self.num_clip, c, h, w).permute(0, 2, 3, 1, 4).reshape(-1, c, h, w * self.num_clip)  # [bs, dim, 6, num_clip*10]
            x = self.TE(x)   # [bs, num_clip, 1024]
        else:
            x = self.avgpool(x)
            x = x.flatten(1)  # [bs * num_clip, 2048]

        x = self.embed_head(x)

        if embed:
            return x

        x = self.dropout(x)
        x = self.cls_fc(x)

        return x, seq_features



if __name__ == "__main__":

    import torch
    input = torch.rand(32, 3, 180, 320)
    model = CAT(use_TE=True, use_SeqAlign=True)
    cls, seq = model(input)
