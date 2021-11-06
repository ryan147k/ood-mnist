#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/10/7 12:59
# DESCRIPTION:
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tensorboardX import SummaryWriter
import random
from PIL import Image
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=True)
parser.add_argument('--ex_num', type=str)
parser.add_argument('--dataset_type', type=int)
parser.add_argument('--train_class', type=int)
parser.add_argument('--test_classes', type=list)

parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epoch_num', type=int, default=200)
parser.add_argument('--print_iter', type=int, default=20)
args = parser.parse_args()

device = None


class RawMNIST(Dataset):
    def __init__(self, root):
        super(RawMNIST, self).__init__()

        self.root = root
        self.size = len(os.listdir(self.root))

        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        for i in range(10):
            img_path = os.path.join(self.root, f"{str(i)}_{self._num2str(index + 1)}.jpg")
            if os.path.exists(img_path):
                break
        label = int(img_path.split('/')[-1].split('_')[0])

        with open(img_path, 'rb') as f:
            img = Image.open(f)
            data = self.transform(img)
            return torch.Tensor(data), torch.LongTensor([label])

    def __len__(self):
        return self.size

    @staticmethod
    def _num2str(num):
        """
        数字转字符串 eg. 123 -> '000123'
        :return:
        """
        s = str(num)
        for _ in range(6 - len(str(num))):
            s = '0' + s
        return s


class ShiftedMNIST(RawMNIST):
    def __init__(self, _class):
        """
        初始化数据集
        """
        assert 0 <= _class < 7

        super(ShiftedMNIST, self).__init__(root=f'./dataset/mnist_shift/{str(_class)}')


class ConditionalMNIST(RawMNIST):
    def __init__(self, _class, train=True):
        """
        初始化数据集
        """
        assert 0 <= _class < 5

        dir = 'train' if train else 'test'
        super(ConditionalMNIST, self).__init__(root=f'./dataset/mnist_correlation/{str(_class)}/{dir}')


class MarginalMNIST(RawMNIST):
    def __init__(self, _class, train=True):
        """
        初始化数据集
        """
        assert 0 <= _class < 4

        dir = 'train' if train else 'test'
        super(MarginalMNIST, self).__init__(root=f'./dataset/mnist_diversity/{str(_class)}/{dir}')


class ClusteredMNIST(RawMNIST):
    def __init__(self, _class):
        assert 0 <= _class < 8

        super(ClusteredMNIST, self).__init__(root=f'./dataset/mnist_clustered/{str(_class)}')


def train(model,
          save_dir: str,
          model_name: str,
          ex_name: str,
          train_dataset,
          val_dataset,
          test_datasets=None):
    # data

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    if test_datasets is None:
        test_loaders = []
    else:
        test_loaders = [DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
                        for dataset in test_datasets]

    # model

    # 多GPU运行
    model = nn.DataParallel(model)
    model = model.to(device)
    print(model.module)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    lr_scheduler = lambda x: 1.0 if x < 30 else 0.8
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)

    # train

    best_val_acc, best_val_iter = 0.0, None  # 记录全局最优信息
    save_model = False

    writer = SummaryWriter('./runs/{}'.format(ex_name))
    iter = 0
    for epoch in range(args.epoch_num):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).squeeze()

            # forward
            y_hat = model(batch_x)
            loss = loss_fn(y_hat, batch_y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update
            optimizer.step()

            # 计算精度
            _, pred = y_hat.max(1)
            num_correct = (pred == batch_y).sum().item()
            acc = num_correct / len(batch_y)

            iter += 1
            if iter % args.print_iter == 0:
                # 打印信息
                train_loss, train_acc = loss.item(), acc
                val_loss, val_acc = val(model, val_loader)
                test_info_list = [val(model, loader) for loader in test_loaders]
                print("\n[INFO] Epoch {} Iter {}:".format(epoch, iter))
                print("\t\t\t\t\t\tTrain: Loss {:.4f}, Accuracy {:.4f}".format(train_loss, train_acc))
                print("\t\t\t\t\t\tVal:   Loss {:.4f}, Accuracy {:.4f}".format(val_loss, val_acc))

                test_acc_dict, test_loss_dict = {}, {}
                for ii, (test_loss, test_acc) in enumerate(test_info_list):
                    print("\t\t\t\t\t\tTest{}: Loss {:.4f}, Accuracy {:.4f}".format(ii, test_loss, test_acc))
                    test_acc_dict[f'test{ii}_acc'] = test_acc
                    test_loss_dict[f'test{ii}_loss'] = test_loss

                acc_value_dict = {'train_acc': train_acc,
                                  'val_acc': val_acc}
                loss_value_dict = {'train_loss': train_loss,
                                   'val_loss': val_loss}
                acc_value_dict.update(
                    test_acc_dict
                )
                loss_value_dict.update(
                    test_loss_dict
                )
                tensorboard_write(writer=writer,
                                  ex_name=ex_name,
                                  mode_name='{} {}'.format(model_name, 'acc'),
                                  value_dict=acc_value_dict,
                                  x_axis=iter)
                tensorboard_write(writer=writer,
                                  ex_name=ex_name,
                                  mode_name='{} {}'.format(model_name, 'loss'),
                                  value_dict=loss_value_dict,
                                  x_axis=iter)

                # 更新全局最优信息
                if val_acc > best_val_acc:
                    best_val_acc, best_val_iter = val_acc, iter
                    save_model = True
                if save_model:
                    # 保存模型
                    torch.save(model.module.state_dict(), os.path.join(save_dir, f'{model_name}_best.pt'))
                    save_model = False

                print("\t best val   acc so far: {:.4} Iter: {}".format(best_val_acc, best_val_iter))

                # torch.save(model.module.state_dict(), os.path.join(save_dir, f'{model_name}_i{iter}.pt')

        scheduler.step()

        # 保存模型
        save_epoch = False
        if save_epoch:
            epoch_save_path = os.path.join(save_dir, 'epoch')
            if not os.path.exists(epoch_save_path):
                os.mkdir(epoch_save_path)
            torch.save(model.module.state_dict(), f'{epoch_save_path}/{model_name}_e{epoch}.pt')


@torch.no_grad()
def val(model, dataloader):
    """
    batch级别的loss & 样本级别的acc
    :param model:
    :param dataloader:
    :return:
    """
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    loss_sum = 0
    correct_sum = 0
    num_x = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).squeeze()

        # forward
        y_hat = model(batch_x)
        loss = loss_fn(y_hat, batch_y)

        # 计算精度
        _, pred = y_hat.max(1)
        num_correct = (pred == batch_y).sum().item()

        num_x += len(batch_x)
        loss_sum += loss.item()
        correct_sum += num_correct

    model.train()

    return loss_sum / len(dataloader), correct_sum / num_x


def tensorboard_write(writer, ex_name, mode_name, value_dict, x_axis):
    """
    tensorboardX 作图
    :param writer:
    :param ex_name:
    :param mode_name: 模型名称+数据 eg. resnet18 acc
    :param value_dict:
    :param x_axis:
    :return:
    """
    writer.add_scalars(main_tag='{}/{}'.format(ex_name, mode_name),
                       tag_scalar_dict=value_dict,
                       global_step=x_axis)


def split_train_val(dataset, split_num=None):
    """
    将dataset分成两个dataset: train & val
    :param split_num:
    :param dataset:
    :return:
    """

    class _Dataset(Dataset):
        def __init__(self, dataset, _train, split_num):
            self.dataset = dataset

            index_list = list(range(len(dataset)))
            random.seed(2)
            random.shuffle(index_list)

            if split_num is None:
                split_num = int(len(dataset) * 9 / 10)
            train_index_list = index_list[:split_num]
            val_index_list = index_list[split_num:]

            self.index_list = train_index_list if _train else val_index_list

        def __getitem__(self, index):
            data, label = self.dataset[self.index_list[index]]
            return data, label

        def __len__(self):
            return len(self.index_list)

        def collate_fn(self, batch):
            return self.dataset.collate_fn(batch)

    return _Dataset(dataset, _train=True, split_num=split_num), \
        _Dataset(dataset, _train=False, split_num=split_num)


class Experiment:
    """
    记录每一次的实验设置
    """

    @staticmethod
    def _mkdir(save_dir):
        """
        如果目录不存在, 则创建
        :param save_dir: 模型检查点保存目录
        :return:
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    @staticmethod
    def _get_dataset(dataset_type, _class, train=True):
        """
        根据 args.dataset_type 来获取 dataset
        :param dataset_type:
        :param _class:
        :return:
        """
        assert 0 <= dataset_type < 4

        _class = int(_class)

        if dataset_type == 0:
            return ShiftedMNIST(_class=_class)
        elif dataset_type == 1:
            return MarginalMNIST(_class=_class, train=train)
        elif dataset_type == 2:
            return ConditionalMNIST(_class=_class, train=train)
        else:
            return ClusteredMNIST(_class=_class)

    @classmethod
    def _ex(cls, model, save_dir, model_name, ex_name, dataset_type, train_class, test_classes):
        """
        模型训练 or 测试
        :param model:
        :param save_dir: 模型检查点保存目录
        :param model_name: 模型名称
        :param ex_name:
        :return:
        """
        model_name = f'd{str(dataset_type)}c{str(train_class)}_{model_name}'

        dataset = cls._get_dataset(dataset_type, _class=train_class)

        train_dataset, val_dataset = split_train_val(dataset)
        if test_classes is None:
            test_classes = []
        test_datasets = [cls._get_dataset(dataset_type, _class=test_class) for test_class in test_classes]

        train(model, save_dir=save_dir, model_name=model_name, ex_name=ex_name,
              train_dataset=train_dataset, val_dataset=val_dataset, test_datasets=test_datasets)

    @classmethod
    def ex1(cls):
        args.batch_size = 1024
        print(args)

        ex_name = 'ex1'
        save_dir = './ckpts/ex1'
        cls._mkdir(save_dir)

        model = models.resnet18(num_classes=10)
        model_name = 'res18'

        return cls._ex(model, save_dir, model_name, ex_name,
                       args.dataset_type, args.train_class, args.test_classes)

    @classmethod
    def ex2(cls):
        args.batch_size = 1024
        print(args)

        ex_name = 'ex2'
        save_dir = './ckpts/ex2'
        cls._mkdir(save_dir)

        model = models.squeezenet1_0(num_classes=10)
        model_name = 'squeezenet1_0'

        return cls._ex(model, save_dir, model_name, ex_name,
                       args.dataset_type, args.train_class, args.test_classes)

    @classmethod
    def ex3(cls):
        args.batch_size = 1024
        print(args)

        ex_name = 'ex3'
        save_dir = './ckpts/ex3'
        cls._mkdir(save_dir)

        model = models.squeezenet1_1(num_classes=10)
        model_name = 'squeezenet1_1'

        return cls._ex(model, save_dir, model_name, ex_name,
                       args.dataset_type, args.train_class, args.test_classes)

    @classmethod
    def ex4(cls):
        args.batch_size = 1024
        print(args)

        ex_name = 'ex4'
        save_dir = './ckpts/ex4'
        cls._mkdir(save_dir)

        model = models.resnet34(num_classes=10)
        model_name = 'resnet34'

        return cls._ex(model, save_dir, model_name, ex_name,
                       args.dataset_type, args.train_class, args.test_classes)

    @classmethod
    def ex5(cls):
        args.batch_size = 1024
        print(args)

        ex_name = 'ex5'
        save_dir = './ckpts/ex5'
        cls._mkdir(save_dir)

        model = models.mobilenet_v3_small(num_classes=10)
        model_name = 'mobilenet_v3_small'

        return cls._ex(model, save_dir, model_name, ex_name,
                       args.dataset_type, args.train_class, args.test_classes)

    @classmethod
    def ex6(cls):
        """
        pretrained model
        :return:
        """
        args.batch_size = 32
        args.epoch = 200
        print(args)

        ex_name = 'ex6'
        save_dir = './ckpts/ex6'
        cls._mkdir(save_dir)

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 10)
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
        model_name = 'res18'

        dataset = ShiftedMNIST(_class=1)
        train_dataset, _ = split_train_val(dataset, 200)
        train(model, save_dir, model_name, ex_name, train_dataset, train_dataset)


def tst():
    import csv

    def _basic_test(model, dataset, batch_size=args.batch_size):
        """
        获取模型在某个测试集上的loss和acc
        :param batch_size:
        :param model:
        :param dataset:
        :return:
        """
        model = nn.parallel.DataParallel(model)
        model.to(device)

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers)
        loss, acc = val(model, loader)
        return loss, acc

    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_list = [models.resnet18(num_classes=10),
                  models.squeezenet1_0(num_classes=10),
                  models.squeezenet1_1(num_classes=10),
                  models.resnet34(num_classes=10),
                  models.mobilenet_v3_small(num_classes=10)]

    res = []

    for i in range(5):
        res.append([])

        ckpt_list = [f'./ckpts/ex1/d2c{i}_res18_best.pt',
                     f'./ckpts/ex2/d2c{i}_squeezenet1_0_best.pt',
                     f'./ckpts/ex3/d2c{i}_squeezenet1_1_best.pt',
                     f'./ckpts/ex4/d2c{i}_resnet34_best.pt',
                     f'./ckpts/ex5/d2c{i}_mobilenet_v3_small_best.pt']

        dataset_ood = ConditionalMNIST(_class=i, train=False)

        for model, ckpt in zip(model_list, ckpt_list):
            model.load_state_dict(torch.load(ckpt))
            _, ood = _basic_test(model, dataset_ood)
            res[-1].append(ood * 100)

    for l in res:
        for n in l:
            print(n, end=', ')
        print()


if __name__ == '__main__':
    if args.debug is not True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ex = getattr(Experiment, f'ex{args.ex_num.strip()}')
        ex()
    else:
        tst()
