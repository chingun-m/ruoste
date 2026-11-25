
import torch
from torch import nn

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

# 2. Convert characters into numbers
chars = sorted(set(text))
to_id = {ch: i for i, ch in enumerate(chars)}
to_char = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([to_id[ch] for ch in text])

# 3. Prepare training examples
context_size = 8

x_train = []
y_train = []

for i in range(context_size, len(data)):
    x_train.append(data[i - context_size:i])
    y_train.append(data[i])

x_train = torch.stack(x_train)
y_train = torch.stack(y_train)

# 4. Build the neural network
vocab_size = len(chars)
embedding_size = 16

model = nn.Sequential(
    nn.Embedding(vocab_size, embedding_size),
    nn.Flatten(),
    nn.Linear(context_size * embedding_size, 64),
    nn.ReLU(),
    nn.Linear(64, vocab_size),
)

# 5. Configure training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. Train the model
for epoch in range(1000):
    logits = model(x_train)
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 7. Generate text by sampling characters
model.eval()

def generate(start, length=100):
    if len(start) < context_size:
        raise ValueError(
            f"Starting text must contain at least {context_size} characters."
        )

    if any(ch not in to_id for ch in start):
        raise ValueError("Starting text contains characters not in the training data.")

    context = start
    result = start

    for _ in range(length):
        input_ids = torch.tensor([
            [to_id[ch] for ch in context[-context_size:]]
        ])

        with torch.no_grad():
            logits = model(input_ids)
            probabilities = torch.softmax(logits[0], dim=0)

        next_id = int(torch.multinomial(probabilities, num_samples=1).item())
        next_ch = to_char[next_id]

        result += next_ch
        context += next_ch

    return result

# 8. Test it
print("\nGenerated text:")
print(generate("hello ru", 100))
