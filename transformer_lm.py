import torch
from torch import nn

torch.manual_seed(42)

# 1. Training data
text = (
    "hello rust. "
    "hello world. "
    "rust is fast. "
    "rust is fun. "
    "rust is powerful. "
    "world is big. "
    "world is beautiful. "
    "coding is fun. "
    "coding with rust is fun. "
    "i love rust. "
    "i love coding. "
) * 20

# 2. Character vocabulary
chars = sorted(set(text))
to_id = {ch: i for i, ch in enumerate(chars)}
to_char = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([to_id[ch] for ch in text])

# 3. Training examples (each position t predicts data[i + t + 1])
context_size = 16
x_train = []
y_train = []

for i in range(len(data) - context_size):
    x_train.append(data[i : i + context_size])
    y_train.append(data[i + 1 : i + context_size + 1])

x_train = torch.stack(x_train)
y_train = torch.stack(y_train)


# 4. Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
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
            attn_mask=mask,
            need_weights=False,
        )

        x = self.norm1(x + attended)
        x = self.norm2(x + self.feed_forward(x))
        return x


# 5. Character-level language model
class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        embed_dim = 32

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_size, embed_dim)

        self.transformer = TransformerBlock(embed_dim, num_heads=4)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

        self.mask: torch.Tensor
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_size, context_size, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def forward(self, token_ids):
        _, sequence_length = token_ids.shape

        positions = torch.arange(
            sequence_length, device=token_ids.device
        )

        x = self.token_embedding(token_ids)
        x = x + self.position_embedding(positions)

        x = self.transformer(x, self.mask[:sequence_length, :sequence_length])

        return self.output_layer(x)


# 6. Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LanguageModel(len(chars)).to(device)
x_train = x_train.to(device)
y_train = y_train.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

for epoch in range(500):
    logits = model(x_train)

    loss = loss_fn(
        logits.reshape(-1, len(chars)),
        y_train.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


# 7. Generate text
model.eval()


def generate(start, length=100):
    if not start or any(ch not in to_id for ch in start):
        raise ValueError("Starting text must use characters in the training data.")

    result = start
    generated = torch.tensor(
        [[to_id[ch] for ch in start[-context_size:]]],
        device=device,
    )

    for _ in range(length):
        with torch.no_grad():
            logits = model(generated[:, -context_size:])
            probabilities = torch.softmax(logits[0, -1], dim=0)

        next_id = torch.multinomial(probabilities, num_samples=1).item()
        result += to_char[int(next_id)]

        generated = torch.cat(
            [generated, torch.tensor([[int(next_id)]], device=device)],
            dim=1,
        )

    return result


print("\nGenerated text:")
print(generate("hello rust", 100))
