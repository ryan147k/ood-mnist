import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def get_input_grad(model, input):
    """
    获取输入的梯度
    :param model:
    :param input:
    :return:
    """
    input.requires_grad = True
    preds = model(input)
    preds, _ = torch.max(preds, 1)
    pred = torch.sum(preds)
    grads_pred = torch.autograd.grad(pred, input)
    return grads_pred[0]


def plot_confusion_matrix(confusion_mat):
    cm = confusion_mat[:10, :10]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


@torch.no_grad()
def get_confusion_matrix(model, loader, device):
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    all_pred = torch.tensor([]).to(device)
    all_label = torch.tensor([]).to(device)

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # forward
        pred = model(batch_x)
        _, pred = torch.max(pred, dim=1)

        all_pred = torch.cat((all_pred, pred), dim=0)
        all_label = torch.cat((all_label, batch_y), dim=0)
    return confusion_matrix(all_label.cpu(), all_pred.cpu())
