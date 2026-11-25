#Self attention: lets positions exchange information.
#Feed forward network: preocesses the information at each postition.
#Residual connections and layer normalization: help training work reliably.
#What the block thing does: attention mixes information between positions, the feed forward network
#processes it, and residual connections help preserve information.
import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim = 16, num_heads = 2):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first = True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x, mask):
        attended, _ = self.attention(
            x, x, x,
            attn_mask = mask,
            need_weights = False,
        )

        x = self.norm1(x + attended)
        processed = self.feed_forward(x)
        x = self.norm2(x + processed)

        return x

#Example input: 4 characters, each represented by 8 numbers
x = torch.randn(1, 8, 16)

#Self attention layer
# attention = nn.MultiheadAttention(
#     embed_dim = 8,
#     num_heads = 2,
#     batch_first = True,
# )

#Block access to future postitions
mask = torch.triu(
    torch.ones(8, 8, dtype=torch.bool),
    diagonal = 1,
)

# output, weights = attention(
#     x, x, x,
#     attn_mask=mask,
#     average_attn_weights = False,
# )

# print("Input shape:", x.shape)
# print("Output shape:", output.shape)
# print("Attention weights shape:", weights.shape)
# print("\nAttention weights:")
# print(weights[0,0])

model = TransformerBlock()
output = model(x, mask)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Transformer block works!")
