import torch, collections, time
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output
from functools import wraps

from utils.visualization import visualize_pca, visualize_tsne

InteractiveShell.ast_node_interactivity = "all"
mask = collections.namedtuple('mask', ('train', 'test'))



def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Time to run function '{}': {:.2f} seconds".format(func.__name__,
                                                                end-start))
        return result
    return wrapper

def train_one_epoch(model, criterion, optimizer, x, y, train_mask=None): 
    model.train()
    out, h = model(**x)
    loss = criterion(out, y) if train_mask is None else criterion(out[train_mask], y[train_mask])
    _, predicted = torch.max(out.detach(), 1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        if train_mask is None:
            length = len(y)
            accuracy = (predicted == y).sum().item() / length
            misclassified = (predicted != y).numpy()
        else:
            length = len(y[train_mask])
            accuracy = (predicted[train_mask] == y[train_mask].detach()).sum().item() / length
            misclassified = (predicted[train_mask] != y[train_mask]).numpy()

    return out, loss.item(), accuracy, misclassified


def test(model, x, y, test_mask=None):  # x is a dictionary
    model.eval()
    with torch.no_grad():
        out, h = model(**x)
        _, predicted = torch.max(out, 1)
        if test_mask is None:
            length = len(y)
            accuracy = (predicted == y).sum().item() / length
        else:
            length = len(y[test_mask])
            accuracy = (predicted[test_mask] == y[test_mask]).sum().item() / length
    return accuracy, predicted[test_mask].numpy()


def plot_acc(train_acc, test_acc=None, xaxis='epochs', yaxis='accuracy',
             title='Accuracy plot'):
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    if test_acc is not None:
        plt.plot(np.arange(len(train_acc)), train_acc, color='red')
        plt.plot(np.arange(len(test_acc)), test_acc, color='blue')
        plt.legend(['train accuracy', 'test accuracy'], loc='upper right')
    else:
        plt.plot(np.arange(len(train_acc)), train_acc, color='red')
        plt.legend(['train accuracy'], loc='upper right')
    plt.title(title)
    plt.tight_layout()
    plt.show()  # show train_acc and test_acc together


def plot_loss(loss, xaxis='epochs', yaxis='loss', title='Loss plot'):
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.plot(np.arange(len(loss)), loss, color='black')
    plt.title(title)
    plt.tight_layout()
    plt.show()


@timethis
def train(epochs, model, criterion, optimizer, x, y, m=mask(None, None),
          plotting=True, scatter_size=30, plotting_freq=5,
          dim_reduction='pca'):
    dim_reduction_dict = {'pca': visualize_pca, 'tsne': visualize_tsne}
    train_acc_list = []
    test_acc_list = []
    loss_list = []
    for epoch in range(epochs):
        out, loss, train_acc, misclassified = train_one_epoch(model,
                                                              criterion,
                                                              optimizer, x, y,
                                                              m.train)
        model.eval()
        test_acc, predictions = test(model, x, y, m.test)
        train_acc_list.append(train_acc)
        loss_list.append(loss)
        test_acc_list.append(test_acc)
        if plotting:
            if epoch % plotting_freq == 0:
                clear_output(wait=True)
                dim_reduction_dict[dim_reduction](out, color=y,
                                                  size=scatter_size,
                                                  epoch=epoch,
                                                  loss=loss)
    if plotting:
        if m == mask(None, None):
            plot_acc(train_acc_list)
        else:
            plot_acc(train_acc_list, test_acc_list)
        plot_loss(loss_list)
    print("Final test accuracy: {:.2f}".format(test_acc_list[-1]))
    return(train_acc_list, test_acc_list, loss_list,
           misclassified, predictions, out)
