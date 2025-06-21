import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        numeric_num_features,
        cat_cardinalities,
        output_dim=7,
        dropout=0.2,
        embed_dim=8
    ):
        super().__init__()
        # 類別型欄位 Embedding
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim=embed_dim) for cardinality in cat_cardinalities
        ])
        emb_total_dim = len(cat_cardinalities)*embed_dim if cat_cardinalities else 0
        input_dim = numeric_num_features + emb_total_dim

        self.fc1 = nn.Linear(input_dim, 256)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc_out = nn.Linear(256, output_dim)
        nn.init.xavier_uniform_(self.fc_out.weight)

        self.relu = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, num_x, cat_x):
        # num_x: (batch, 數值欄位數)；cat_x: (batch, 類別欄位數)
        emb = [emb_layer(cat_x[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1) if emb else None
        if emb is not None:
            x = torch.cat([num_x, emb], dim=1)
        else:
            x = num_x

        out1 = self.drop(self.relu(self.bn1(self.fc1(x))))
        out2 = self.drop(self.relu(self.bn2(self.fc2(out1))))
        out2 = out2 + out1
        prob = self.fc_out(out2)
        return prob