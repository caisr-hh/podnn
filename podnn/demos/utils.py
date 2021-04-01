import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch

#-----------------------------------------------------------------------

def plot_bounday_tensorflow(model,n_models,x_train,y_train,x_test,y_test):

    w = model.layers[-1].weights[0]
    idx = np.where(w < 0)[0]
    colors = ['black' if l == 0 else 'purple' for l in y_train]

    plot_step = 0.02
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    input = np.c_[xx.ravel(), yy.ravel()]
    input = tf.convert_to_tensor(input)
    _, Z = model(input)


    n_rows = int(np.floor(np.sqrt(n_models)))
    n_cols = int(np.floor(np.sqrt(n_models)))
    fig, ax = plt.subplots(nrows=n_rows,ncols=n_cols)

    model_number = 0
    for r in range(n_rows):

        for c in range(n_cols):

            Z_current = np.round(np.reshape(Z[:,model_number], xx.shape))
            if model_number in idx:
                Z_current = 1 - Z_current
            boundary = ax[r,c].contourf(xx, yy, Z_current, cmap=plt.cm.RdYlBu)
            ax[r,c].scatter(x_train[:, 0], x_train[:, 1], c=colors,s=5)
            model_number += 1

    fig.colorbar(boundary, ax=ax, shrink=0.6)
    ax[0, 0].set_title('Aggregation layer decision boundaries')
    ax[0,1].set_title('Aggregation layer decision boundaries')
    plt.axis('off')
    plt.show(block=False)
    plt.axis('off')
    plt.show()

    plt.figure()
    input = np.c_[xx.ravel(), yy.ravel()]
    Z = tf.convert_to_tensor(input)

    preds,_ = model(Z)
    preds = np.reshape(preds, xx.shape)
    boundary = plt.contourf(xx, yy, preds, cmap=plt.cm.RdYlBu)
    plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
    plt.show()


def plot_bounday_torch(model,n_models,x_train,y_train,x_test,y_test):


    w = model[-2].weight[0]
    idx = np.where(w < 0)[0]
    colors = ['black' if l == 0 else 'purple' for l in y_train]

    plot_step = 0.02
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    input = np.c_[xx.ravel(), yy.ravel()]
    Z = torch.from_numpy(input).float()
    counter = 0
    for layer in model:
        Z = layer(Z)
        if counter==3:
            break
        counter += 1
    Z = torch.detach(Z)

    n_rows = int(np.floor(np.sqrt(n_models)))
    n_cols = int(np.floor(np.sqrt(n_models)))
    fig, ax = plt.subplots(nrows=n_rows,ncols=n_cols)

    model_number = 0
    for r in range(n_rows):

        for c in range(n_cols):

            Z_current = np.round(np.reshape(Z[:,model_number], xx.shape))
            if model_number in idx:
                Z_current = 1 - Z_current
            boundary = ax[r,c].contourf(xx, yy, Z_current, cmap=plt.cm.RdYlBu)
            ax[r,c].scatter(x_train[:, 0], x_train[:, 1], c=colors,s=5)
            model_number += 1

    fig.colorbar(boundary, ax=ax, shrink=0.6)
    ax[0, 0].set_title('Aggregation layer decision boundaries')
    ax[0,1].set_title('Aggregation layer decision boundaries')
    plt.axis('off')
    plt.show(block=False)
    plt.axis('off')
    plt.show()

    plt.figure()
    input = np.c_[xx.ravel(), yy.ravel()]
    Z = torch.from_numpy(input).float()

    preds = torch.detach(model(Z))
    preds = np.reshape(preds, xx.shape)
    boundary = plt.contourf(xx, yy, preds, cmap=plt.cm.RdYlBu)
    plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
    plt.show()
