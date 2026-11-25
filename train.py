import torch
from torch import nn

#Example training: y=2x+1
x_train = torch.tensor([[2.0],[4.0],[6.0],[7.0],[1.0],[3.0],[10.0],[8.0],[9.0],[11.0],[12.0],[13.0],[14.0],[15.0],[16.0]])
y_train = torch.tensor([[5.0],[9.0],[13.0],[15],[3.0],[7.0],[21.0],[17.0],[19.0],[23.0],[25.0],[27.0],[29.0],[31.0],[33.0]])

#Neural network
#nn.Linear(1, 1) creates a tiny neural network wth one input and one output.
#nternally, it learns two numbers: a weight W and a bias b.
#y = Wx+b
#Initially these numbers are random.
model = nn.Linear(1, 1)

#How the model learns
#nn.MSELoss() measures the average squared difference between predictions and
#correct answers.
#If the model predicts 6 when the answer is 9, that prediction contributes an error
#of (6-9)^2 = 9 to the loss.
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Trainging loop
for epoch in range(1000):
    predictions = model(x_train)
    loss = loss_fn(predictions, y_train)

    optimizer.zero_grad()
    #loss.backward() calculates gradients that tell the optimizer how the weight and
    #bias affect the loss. It doesn't update the parameters itself. It calculates the information
    #needed to update them.
    loss.backward()
    #optimizer.step() adjusts the weight and bias using the calculated gradieants.
    #We repeat this process 1,000 times, giving the model opportunities to improve.
    optimizer.step()

    if(epoch + 1) & 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item(): .6f}")

#Test on a value it hasn't seen
with torch.no_grad():
    result = model(torch.tensor([[5.0]]))
    print("Prediction for x=5: ", result.item())

    print("Learned weight:", model.weight.item())
    print("Learned bias:", model.bias.item())
