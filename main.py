#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/10/7 12:59
# DESCRIPTION:
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from model import *
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
parser.add_argument('--dataset_id', type=int)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--print_iter', type=int, default=20)
args = parser.parse_args()


class RawMNIST(Dataset):
    def __init__(self, root):
        super(RawMNIST, self).__init__()

        self.root = root
        self.size = len(os.listdir(self.root))

        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        for i in range(10):
            img_path = f"{self.root}/{str(i)}_{self._num2str(index + 1)}.jpg"
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


class CorrelationMNIST(RawMNIST):
    def __init__(self, _class):
        """
        初始化数据集
        """
        assert 0 <= _class < 5

        super(CorrelationMNIST, self).__init__(root=f'./dataset/mnist_correlation/{str(_class)}')


class DiversityMNIST(RawMNIST):
    def __init__(self, _class):
        """
        初始化数据集
        :param data_dir: 数据集目录
        :param dataset_name:  数据集名称 eg. train validation test1
        """
        assert 0 <= _class < 5

        super(DiversityMNIST, self).__init__(root=f'./dataset/mnist_diversity/{str(_class)}')


class ClusteredMNIST(RawMNIST):
    def __init__(self, _class):
        assert 0 <= _class < 5

        super(ClusteredMNIST, self).__init__(root=f'./dataset/mnist_clustered/{str(_class)}')


def train(model,
          save_path: str,
          ex_name: str,
          train_dataset,
          val_dataset,
          test_datasets=None):
    # data

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=8)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=8)
    if test_datasets is None:
        test_loaders = []
    else:
        test_loaders = [DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=2)
                        for dataset in test_datasets]

    # model

    # 多GPU运行
    model = nn.DataParallel(model)
    model = model.to(device)
    print(model.module)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    lr_scheduler = lambda x: 1.0 if x < 15 else 0.5
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)

    # train

    best_val_acc, best_val_iter = 0.0, None  # 记录全局最优信息
    save_model = False

    writer = SummaryWriter()
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
                print("\n[INFO] Epoch {} Iter {}:\n \
                            \tTrain: Loss {:.4f}, Accuracy {:.4f}\n \
                            \tVal:   Loss {:.4f}, Accuracy {:.4f}".format(epoch + 1, iter,
                                                                          train_loss, train_acc,
                                                                          val_loss, val_acc))
                for ii, (test_loss, test_acc) in enumerate(test_info_list):
                    print("\tTest{}: Loss {:.4f}, Accuracy {:.4f}".format(ii, test_loss, test_acc))

                tensorboard_write(writer=writer,
                                  ex_name=ex_name,
                                  mode_name='{} {}'.format(save_path.split('/')[-1], 'acc'),
                                  value_dict={'train_acc': train_acc,
                                              'val_acc': val_acc},
                                  x_axis=iter)
                tensorboard_write(writer=writer,
                                  ex_name=ex_name,
                                  mode_name='{} {}'.format(save_path.split('/')[-1], 'loss'),
                                  value_dict={'train_loss': train_loss,
                                              'val_loss': val_loss},
                                  x_axis=iter)

                # 更新全局最优信息
                if val_acc > best_val_acc:
                    best_val_acc, best_val_iter = val_acc, iter
                    save_model = True
                if save_model:
                    # 保存模型
                    torch.save(model.module.state_dict(), '{}_best.pt'.format(save_path))
                    save_model = False

                print("\t best val   acc so far: {:.4} Iter: {}".format(best_val_acc, best_val_iter))

                # torch.save(model.module.state_dict(), '{}_i{}.pt'.format(save_path, iter))

        # 保存模型
        # torch.save(model.module.state_dict(), '{}_e{}.pt'.format(save_path, epoch))
        scheduler.step()


@torch.no_grad()
def val(model, dataloader):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    loss_sum = 0
    acc_sum = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).squeeze()

        # forward
        y_hat = model(batch_x)
        loss = loss_fn(y_hat, batch_y)

        loss_sum += loss.item()
        # 计算精度
        _, pred = y_hat.max(1)
        num_correct = (pred == batch_y).sum().item()
        acc = num_correct / len(batch_y)
        acc_sum += acc

    model.train()

    return loss_sum / len(dataloader), acc_sum / len(dataloader)


def plot(img):
    # (c, h, w) -> (h, w, c)
    img = np.transpose(img, (1, 2, 0))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def tensorboard_write(writer, ex_name: str, mode_name, value_dict, x_axis):
    """
    tensorboardX 作图
    :param writer:
    :param ex_name: 实验名称
    :param mode_name: 模型名称+数据 eg. resnet18 acc
    :param value_dict:
    :param x_axis:
    :return:
    """
    writer.add_scalars(main_tag='{}/{}'.format(ex_name, mode_name),
                       tag_scalar_dict=value_dict,
                       global_step=x_axis)


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
    def _split_train_val(dataset):
        """
        将dataset分成两个dataset: train & val
        :param dataset:
        :return:
        """

        class _Dataset(Dataset):
            def __init__(self, dataset, _train=True):
                self.dataset = dataset

                index_list = list(range(len(dataset)))
                random.seed(2)
                random.shuffle(index_list)

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

        return _Dataset(dataset, _train=True), _Dataset(dataset, _train=False)

    @staticmethod
    def _get_dataset(dataset_id, _class):
        """
        根据 args.dataset_id 来获取 dataset
        :param dataset_id:
        :param _class:
        :return:
        """
        assert 0 <= dataset_id < 3

        if dataset_id == 0:
            return ClusteredMNIST(_class=_class)
        elif dataset_id == 1:
            return CorrelationMNIST(_class=_class)
        else:
            return DiversityMNIST(_class=_class)

    @staticmethod
    def _basic_test(model, dataset):
        """
        获取模型在某个测试集上的loss和acc
        :param model:
        :param dataset:
        :return:
        """
        model = nn.parallel.DataParallel(model)
        model.to(device)

        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=10)
        loss, acc = val(model, loader)
        return loss, acc

    @classmethod
    def _ex_train(cls, model, save_path, ex_name, train_dataset, val_dataset, test_dataset=None):
        """
        模型训练
        :param model:
        :param save_path:
        :param ex_name:
        :param train_dataset:
        :param val_dataset:
        :param test_dataset:
        :return:
        """
        train(model, save_path=save_path, ex_name=ex_name,
              train_dataset=train_dataset, val_dataset=val_dataset, test_datasets=test_dataset)

    @classmethod
    def _ex_test(cls, model, save_path, dataset_id):
        """
        模型测试：返回模型在某种分布迁移下的测试准确率列表
        :param model:
        :param save_path
        :param dataset_id:
        :return:
        """
        model.load_state_dict(torch.load(f'{save_path}_best.pt'))
        acc_list = []

        num_cluster = 5

        for i in range(num_cluster):
            dataset = cls._get_dataset(dataset_id, _class=i)

            # i == 0 时,相当于是训练数据,所以只取验证集出来
            if i == 0:
                _, dataset = cls._split_train_val(dataset)

            _, acc = cls._basic_test(model, dataset)
            acc_list.append(acc)

        return acc_list

    @classmethod
    def _ex(cls, _train, model, save_dir, model_name, ex_name, dataset_id):
        """
        模型训练 or 测试
        :param _train: 训练 or 测试
        :param model:
        :param save_dir: 模型检查点保存模型
        :param model_name: 模型名称
        :param ex_name:
        :return:
        """
        model_name = f'd{str(args.dataset_id)}_{model_name}'
        save_path = os.path.join(save_dir, model_name)

        if _train:
            dataset = cls._get_dataset(args.dataset_id, _class=0)
            train_dataset, val_dataset = cls._split_train_val(dataset)

            cls._ex_train(model, save_path, ex_name, train_dataset, val_dataset)
        else:
            return cls._ex_test(model, save_path, dataset_id)

    @classmethod
    def ex1(cls, _train=False):
        args.batch_size = 1024
        print(args)

        ex_name = 'ex1'
        save_dir = './ckpts/ex1'
        cls._mkdir(save_dir)

        model = models.resnet18(num_classes=10)
        model_name = 'res18'

        return cls._ex(_train, model, save_dir, model_name, ex_name, args.dataset_id)

    @classmethod
    def ex2(cls, _train=False):
        args.batch_size = 1024
        print(args)

        ex_name = 'ex2'
        save_dir = './ckpts/ex2'
        cls._mkdir(save_dir)

        model = models.squeezenet1_0(num_classes=10)
        model_name = 'squeezenet1_0'

        return cls._ex(_train, model, save_dir, model_name, ex_name, args.dataset_id)

    @classmethod
    def ex3(cls, _train=False):
        args.batch_size = 1024
        print(args)

        ex_name = 'ex3'
        save_dir = './ckpts/ex3'
        cls._mkdir(save_dir)

        model = models.squeezenet1_1(num_classes=10)
        model_name = 'squeezenet1_1'

        return cls._ex(_train, model, save_dir, model_name, ex_name, args.dataset_id)

    @classmethod
    def ex4(cls, _train=False):
        print(args)

        ex_name = 'ex4'
        save_dir = './ckpts/ex4'
        cls._mkdir(save_dir)

        model = models.mobilenet_v2(num_classes=10)
        model_name = 'mobilenet_v2'

        return cls._ex(_train, model, save_dir, model_name, ex_name, args.dataset_id)

    @classmethod
    def ex5(cls, _train=False):
        print(args)

        ex_name = 'ex5'
        save_dir = './ckpts/ex5'
        cls._mkdir(save_dir)

        model = models.mobilenet_v2(num_classes=10)
        model_name = 'mobilenet_v2'

        return cls._ex(_train, model, save_dir, model_name, ex_name, args.dataset_id)

    @classmethod
    def test_(cls):
        """
        获取整体测试结果
        :return:
        """
        # TODO: 修改代码
        res = []

        for dataset_id in range(3):
            args.dataset_id = dataset_id

            acc_list = []

            markers = ['-s', '-o', '-*', '-^', '-D', '-p']
            models = ['ResNet18', 'AlexNet', 'Vgg11', 'DensNet121', 'SqueezeNet', 'ResNext50']
            titles = ['Distribution OOD', 'Correlation OOD', 'Diversity OOD']
            for i, ex_num in enumerate([1, 2, 3, 4, 5, 6]):
                args.ex_num = str(ex_num)
                ex_ = getattr(cls, f'ex{args.ex_num}')
                acc = ex_(False)

                acc_list.append(acc)
                plt.plot(range(len(acc)), acc, markers[i], ms=6, label=models[i], lw=0.4)
            plt.xlabel('OOD Data')
            plt.ylabel('Accuracy')
            plt.title(titles[dataset_id])
            plt.legend()
            plt.show()

            res.append(acc_list)

        pickle.dump(res, open('./count/res_test.pkl', 'wb'))


if __name__ == '__main__':
    _train = True
    if args.debug is True:
        args.ex_num = '3'
        args.dataset_id = 2
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        _train = False

    ex = getattr(Experiment, f'ex{args.ex_num.strip()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ex(_train)
