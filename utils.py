import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from main import ShiftedMNIST


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
    plt.imshow(cm, interpolation='nearest')
    # plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('scatter.pdf', pad_inches=0.0, dpi=600)
    plt.show()


@torch.no_grad()
def get_confusion_matrix(model, loader, device):
    model.eval()
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    all_pred = torch.tensor([]).to(device)
    all_label = torch.tensor([]).to(device)

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).squeeze()

        # forward
        y_hat = model(batch_x)
        _, pred = y_hat.max(1)

        all_pred = torch.cat((all_pred, pred), dim=0)
        all_label = torch.cat((all_label, batch_y), dim=0)

    num_correct = torch.sum(all_pred == all_label).item()
    acc = num_correct / len(all_label)
    print(acc)

    return confusion_matrix(all_label.cpu(), all_pred.cpu())


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet18(num_classes=10)
    model.load_state_dict(torch.load('./ckpts/ex1/d0c2_res18_best.pt'))
    loader = DataLoader(ShiftedMNIST(_class=5), batch_size=128, num_workers=3)
    plot_confusion_matrix(get_confusion_matrix(model, loader, device))
