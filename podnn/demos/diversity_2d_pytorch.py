import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import podnn_torch
import utils


n_samples = 500
X, y = make_circles(noise=0.3, random_state=17, n_samples=n_samples,factor=0.2)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


X_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train.reshape(-1,1)).float()
X_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test.reshape(-1,1)).float()



unit_model = torch.nn.Sequential(
    nn.Linear(in_features=2, out_features=12),
    nn.ELU(),
    nn.Linear(in_features=12,out_features=10),
)


model = torch.nn.Sequential(

    podnn_torch.InputLayer(n_models=8),
    podnn_torch.ParallelLayer(unit_model),
    podnn_torch.OrthogonalLayer1D(),
    podnn_torch.AggregationLayer(stride=2,input_dim=10),
    nn.Linear(in_features=podnn_torch.agg_out_dim,out_features=1),
    torch.nn.Sigmoid()
)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

for t in range(200):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X_train)

    # Compute and print loss
    loss = criterion(y_pred, y_train)
    train_acc = accuracy_score(y_train,np.round(torch.detach(y_pred)))
    if t % 10 == 0:
        print('epoch:' + str(t) + '   train loss'     + str(loss.item()))
        print('epoch:' + str(t) + '   train accuracy' + str(train_acc))

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

preds_test = model(X_test)
test_acc = accuracy_score(y_test,np.round(torch.detach(preds_test)))
print('=======> test accuracy=' + str(test_acc))

utils.plot_bounday_torch(model,4,x_train,y_train,x_test,y_test)