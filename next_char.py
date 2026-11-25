# nn.Embedding(vocab_size, 16)
# The embedding gives each character a learnable vector of 16 numbers. Then the linear
# layer uses that vector to produce scores for every possible nex character.
# During training, the model adjusts these parameters to reduce the overall prediction error.
# However, our model only receives the current character's ID. Every time it sees l, it gets
# the same input vector, regardless of whether the character came from hello or world.
# It cannot distinguish these contexts:
# * hel -> the next character is l
# * wor -> the next character is l
# * world ending in l -> the next character is a space
# Because it only sees the final l, it cannot tell which of those situations it is in.
# * Why it isn't choosing d?
#   The model is trained on all the character transitions simultaniously. It learns a
#   compromise between the possible next character after l.
# * Also the code uses probabilities[0].argmax()
#   argmax() chooses whichever character has the highest predicted score. It doesn't
#   randomly select a character according to its probability.
#
#
import torch
from torch import nn

# 1. Our tiny training dataset
text = "hello world. hello rust world. hello world. hello world rust."

# 2. Build a vocabulary of unique characters
chars = sorted(set(text))
to_id = {ch: i for i, ch in enumerate(chars)}
to_char = {i: ch for ch, i in to_id.items()}

# Convert characters into numbers
data = torch.tensor([to_id[ch] for ch in text])



# 3. Create training examples:
#    Input: one character
#    Target: the next character
x_train = data[:-1]
y_train = data[1:]

# 4. Our neural network
vocab_size = len(chars)
model = nn.Sequential(
    nn.Embedding(vocab_size, 16),
    nn.Linear(16, vocab_size),
)

# 5. Training setup
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. Training loop
for epoch in range(1000):
    logits = model(x_train)
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 7. Test a prediction
model.eval()

with torch.no_grad():
    ch = "l"
    input_id = torch.tensor([to_id[ch]])
    logits = model(input_id)

    probabilities = torch.softmax(logits, dim=1)
    next_id = int(probabilities[0].argmax().item())

    print(f"\nInput character: {ch!r}")
    print(f"Predicted next character: {to_char[next_id]!r}")
