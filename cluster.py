#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/9/27 9:32
# DESCRIPTION:
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
import random
from PIL import Image
from collections import Counter


class RawMNIST(Dataset):
    """
    根据_class选出某个类别的数字
    """
    def __init__(self, _class, root='./dataset/mnist_shift/0'):
        super(RawMNIST, self).__init__()

        self._class = _class
        self.root = root
        self.imgs = []
        for filename in os.listdir(self.root):
            if f'{str(_class)}_' in filename:
                self.imgs.append(filename)

        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        img_path = f'{self.root}/{self.imgs[index]}'

        with open(img_path, 'rb') as f:
            img = Image.open(f)
            data = self.transform(img)
            return torch.Tensor(data), torch.LongTensor([self._class])

    def __len__(self):
        return len(self.imgs)

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


class ClusterMNIST:
    encoder = tv.models.resnet50(pretrained=True)

    num_class = 10
    num_cluster = 8

    @classmethod
    def t(cls):
        """
        For test
        :return:
        """
        a = cls._get_coding(0)

    @classmethod
    def _get_coding(cls, data_class):
        def _hook(module, input, output):
            hook_res.append(input[0].cpu().detach())

        def _forward(model, dataset):
            with torch.no_grad():
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                loader = DataLoader(dataset, batch_size=128, num_workers=10)
                for data, _ in loader:
                    data = data.to(device)
                    model(data)  # TODO: batch_size太大, 直接放内存放不下

        mnist = RawMNIST(_class=data_class)

        hook_res = []  # 用于接收钩子函数的输出
        cls.encoder.fc.register_forward_hook(_hook)
        _forward(cls.encoder, mnist)

        coding = torch.cat(hook_res, dim=0)

        return coding

    @classmethod
    def plot_pca_coding(cls):
        coding_list = []
        label_list = []
        for i in range(cls.num_class):
            coding = cls._get_coding(i)
            label = torch.zeros(len(coding)) + i
            coding_list.append(coding)
            label_list.append(label)
        coding = torch.cat(coding_list, dim=0)
        label = torch.cat(label_list, dim=0)

        coding = PCA(n_components=2, random_state=2).fit_transform(coding)

        index_list = range(len(coding))
        random.seed(2)
        index_list = random.sample(index_list, 1000)
        coding = coding[index_list]
        label = label[index_list]

        label = label.numpy().tolist()
        plt.scatter(coding[:, 0], coding[:, 1], c=label, cmap='rainbow')
        plt.show()

    @classmethod
    def coding2pkl(cls):
        coding_list = []
        for data_class in range(cls.num_class):
            coding = cls._get_coding(data_class)
            coding_list.append(coding)
        pickle.dump(coding_list, open('./count/coding_list.pkl', 'wb'))

    @classmethod
    def kmeans2pkl(cls, random_state=2):
        coding_list = pickle.load(open('./count/coding_list.pkl', 'rb'))
        cluster_list = []
        for data_class in tqdm(range(cls.num_class)):
            coding = coding_list[data_class]
            cluster = KMeans(n_clusters=cls.num_cluster, random_state=random_state).fit(coding)
            cluster_list.append(cluster)
        pickle.dump(cluster_list, open('./count/kmeans_list.pkl', 'wb'))

    @classmethod
    def print_kmeans_info(cls):
        kmeans_list = pickle.load(open('./count/kmeans_list.pkl', 'rb'))
        for i in range(cls.num_class):
            kmeans = kmeans_list[i]
            count = Counter(kmeans.labels_)
            print(count)

    @classmethod
    def _sort(cls):
        """
        根据kmeans聚类结果, 对每个data_class的聚类按照 聚类大小 排序
        :return: res[i][j] eg. res[0][0]=3 数字0聚类后最大的聚类是标号3的簇
        """
        res = []
        kmeans_list = pickle.load(open('./count/kmeans_list.pkl', 'rb'))
        for i in range(cls.num_class):
            kmeans = kmeans_list[i]
            count = Counter(kmeans.labels_)
            keys, _ = zip(*count.most_common())
            res.append(keys)
        return res

    @classmethod
    def mnist2file(cls):
        def _num2str(num):
            """
            数字转字符串 eg. 123 -> '000123'
            :return:
            """
            s = str(num)
            for _ in range(6 - len(str(num))):
                s = '0' + s
            return s

        root = './dataset/mnist_clustered'
        ranks = cls._sort()
        kmeans_list = pickle.load(open('./count/kmeans_list.pkl', 'rb'))

        for num_cluster in tqdm(range(cls.num_cluster)):
            data_dir = f'{root}/{str(num_cluster)}'
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)

            count = 0
            for data_class in range(cls.num_class):
                kmeans = kmeans_list[data_class]
                rk = ranks[data_class][num_cluster]  # 对于dataclass这个类别, 放到num_cluster文件夹里面的应该是第rk个聚类

                dataset = RawMNIST(_class=data_class)
                for i in range(len(dataset)):

                    if kmeans.labels_[i] == rk:
                        img, label = dataset[i]
                        img = transforms.ToPILImage()(img)
                        name = f'{str(int(label))}_{_num2str(count + 1)}.jpg'
                        img.save(os.path.join(root, f'{str(num_cluster)}/{name}'))
                        count += 1
            print(count)


class ColorMNIST:
    """
    生成数据集: 每张图片的像素值在[0, 1]之间, 大小为[batch_size, height, width, channel]
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='../dataset/',
                             transform=transform,
                             train=True,
                             download=True)
    loader = DataLoader(dataset, batch_size=len(dataset))
    data, labels = next(loader.__iter__())

    @staticmethod
    def _color_img(img, c: str):
        """
        图片着色
        :param img: 灰度图 shape为 [height, width]
        :return:
        """
        assert img.ndim == 2
        assert c in ['r', 'r', 'y', 'y', 'g', 'g', 'b', 'b']

        dtype = img.dtype
        h, w = img.shape
        arr = np.reshape(img, [h, w, 1])
        if c == 'r' or c == 'r':
            arr = np.concatenate([arr,
                                  np.zeros((h, w, 1), dtype=dtype),
                                  np.zeros((h, w, 1), dtype=dtype)], axis=2)
        elif c == 'y' or c == 'y':
            arr = np.concatenate([arr,
                                  arr,
                                  np.zeros((h, w, 1), dtype=dtype)], axis=2)
        elif c == 'g' or c == 'g':
            arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                                  arr,
                                  np.zeros((h, w, 1), dtype=dtype)], axis=2)
        elif c == 'b' or c == 'b':
            arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                                  np.zeros((h, w, 1), dtype=dtype),
                                  arr], axis=2)

        return arr

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

    @classmethod
    def _imgs2file(cls, imgs, color, labels, data_dir):
        """
        将图片上色并输出到文件
        :param imgs:
        :param color:
        :param labels:
        :param data_dir:
        :return:
        """
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        imgs = np.array(imgs).squeeze()
        color = np.array(color).squeeze()
        labels = np.array(labels).squeeze()

        for i, (img, c, label) in enumerate(zip(imgs, color, labels)):
            img = cls._color_img(img, c)
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img)
            img = transforms.ToPILImage()(img)

            name = f'{str(int(label))}_{cls._num2str(i + 1)}.jpg'
            img.save(f'{data_dir}/{name}')

    @classmethod
    def mnist_shift2file(cls):
        root = './dataset/mnist_shift'

        data = cls.data.squeeze()
        labels = cls.labels

        colors = []
        for label in range(10):
            len_c = len(labels[labels == label])
            len1_4 = int(len_c / 4)
            color = ['r'] * len1_4 + ['y'] * len1_4 + ['g'] * len1_4 + ['b'] * (len_c - len1_4 * 3)

            random.seed(2)
            random.shuffle(color)
            colors.append(np.array(color))

        # origin

        data_dir = f'{root}/0'
        imgs, img_colors, img_labels = None, None, None
        for label in range(10):
            data_c = data[labels == label].numpy()
            color_c = colors[label]
            labels_c = np.array([label] * len(data_c))

            imgs = data_c if imgs is None \
                else np.concatenate((imgs, data_c), axis=0)
            img_colors = color_c if img_colors is None \
                else np.concatenate((img_colors, color_c), axis=0)
            img_labels = labels_c if img_labels is None \
                else np.concatenate((img_labels, labels_c), axis=0)

        cls._imgs2file(imgs, img_colors, img_labels, data_dir)

        # marginal

        # 0-9 r&y
        data_dir = f'{root}/1'
        imgs, img_colors, img_labels = None, None, None
        for label in range(10):
            color = colors[label]
            data_c = data[labels == label]  # 某个类别的数据

            index = np.array(color == 'r') + np.array(color == 'y')

            data_c = data_c[index]
            color_c = color[index]
            labels_c = [label] * len(data_c)

            imgs = data_c if imgs is None \
                else np.concatenate((imgs, data_c), axis=0)
            img_colors = color_c if img_colors is None \
                else np.concatenate((img_colors, color_c), axis=0)
            img_labels = labels_c if img_labels is None \
                else np.concatenate((img_labels, labels_c), axis=0)

        cls._imgs2file(imgs, img_colors, img_labels, data_dir)

        # 0-9 g&b
        data_dir = f'{root}/4'
        imgs, img_colors, img_labels = None, None, None
        for label in range(10):
            color = colors[label]
            data_c = data[labels == label]  # 某个类别的数据

            index = np.array(color == 'g') + np.array(color == 'b')

            data_c = data_c[index]
            color_c = color[index]
            labels_c = [label] * len(data_c)

            imgs = data_c if imgs is None \
                else np.concatenate((imgs, data_c), axis=0)
            img_colors = color_c if img_colors is None \
                else np.concatenate((img_colors, color_c), axis=0)
            img_labels = labels_c if img_labels is None \
                else np.concatenate((img_labels, labels_c), axis=0)

        cls._imgs2file(imgs, img_colors, img_labels, data_dir)

        # conditional

        # 0-4 r&y 5-9 g&b
        data_dir = f'{root}/2'
        imgs, img_colors, img_labels = None, None, None
        for label in range(5):
            color = colors[label]
            data_c = data[labels == label]  # 某个类别的数据

            index = np.array(color == 'r') + np.array(color == 'y')

            data_c = data_c[index]
            color_c = color[index]
            labels_c = [label] * len(data_c)

            imgs = data_c if imgs is None \
                else np.concatenate((imgs, data_c), axis=0)
            img_colors = color_c if img_colors is None \
                else np.concatenate((img_colors, color_c), axis=0)
            img_labels = labels_c if img_labels is None \
                else np.concatenate((img_labels, labels_c), axis=0)

        for label in range(5, 10):
            color = colors[label]
            data_c = data[labels == label]  # 某个类别的数据

            index = np.array(color == 'g') + np.array(color == 'b')

            data_c = data_c[index]
            color_c = color[index]
            labels_c = [label] * len(data_c)

            imgs = data_c if imgs is None \
                else np.concatenate((imgs, data_c), axis=0)
            img_colors = color_c if img_colors is None \
                else np.concatenate((img_colors, color_c), axis=0)
            img_labels = labels_c if img_labels is None \
                else np.concatenate((img_labels, labels_c), axis=0)

        cls._imgs2file(imgs, img_colors, img_labels, data_dir)

        # 0-4 g&b 5-9 r&y
        data_dir = f'{root}/5'
        imgs, img_colors, img_labels = None, None, None
        for label in range(5):
            color = colors[label]
            data_c = data[labels == label]  # 某个类别的数据

            index = np.array(color == 'g') + np.array(color == 'b')

            data_c = data_c[index]
            color_c = color[index]
            labels_c = [label] * len(data_c)

            imgs = data_c if imgs is None \
                else np.concatenate((imgs, data_c), axis=0)
            img_colors = color_c if img_colors is None \
                else np.concatenate((img_colors, color_c), axis=0)
            img_labels = labels_c if img_labels is None \
                else np.concatenate((img_labels, labels_c), axis=0)

        for label in range(5, 10):
            color = colors[label]
            data_c = data[labels == label]  # 某个类别的数据

            index = np.array(color == 'r') + np.array(color == 'y')

            data_c = data_c[index]
            color_c = color[index]
            labels_c = [label] * len(data_c)

            imgs = data_c if imgs is None \
                else np.concatenate((imgs, data_c), axis=0)
            img_colors = color_c if img_colors is None \
                else np.concatenate((img_colors, color_c), axis=0)
            img_labels = labels_c if img_labels is None \
                else np.concatenate((img_labels, labels_c), axis=0)

        cls._imgs2file(imgs, img_colors, img_labels, data_dir)

        # joint

        # 0-4 r&y 5-9 y&g
        data_dir = f'{root}/3'
        imgs, img_colors, img_labels = None, None, None
        for label in range(5):
            color = colors[label]
            data_c = data[labels == label]  # 某个类别的数据

            index = np.array(color == 'r') + np.array(color == 'y')

            data_c = data_c[index]
            color_c = color[index]
            labels_c = [label] * len(data_c)

            imgs = data_c if imgs is None \
                else np.concatenate((imgs, data_c), axis=0)
            img_colors = color_c if img_colors is None \
                else np.concatenate((img_colors, color_c), axis=0)
            img_labels = labels_c if img_labels is None \
                else np.concatenate((img_labels, labels_c), axis=0)

        for label in range(5, 10):
            color = colors[label]
            data_c = data[labels == label]  # 某个类别的数据

            index = np.array(color == 'y') + np.array(color == 'g')

            data_c = data_c[index]
            color_c = color[index]
            labels_c = [label] * len(data_c)

            imgs = data_c if imgs is None \
                else np.concatenate((imgs, data_c), axis=0)
            img_colors = color_c if img_colors is None \
                else np.concatenate((img_colors, color_c), axis=0)
            img_labels = labels_c if img_labels is None \
                else np.concatenate((img_labels, labels_c), axis=0)

        cls._imgs2file(imgs, img_colors, img_labels, data_dir)

        # 0-4 g&b 5-9 r&b
        data_dir = f'{root}/6'
        imgs, img_colors, img_labels = None, None, None
        for label in range(5):
            color = colors[label]
            data_c = data[labels == label]  # 某个类别的数据

            index = np.array(color == 'g') + np.array(color == 'b')

            data_c = data_c[index]
            color_c = color[index]
            labels_c = [label] * len(data_c)

            imgs = data_c if imgs is None \
                else np.concatenate((imgs, data_c), axis=0)
            img_colors = color_c if img_colors is None \
                else np.concatenate((img_colors, color_c), axis=0)
            img_labels = labels_c if img_labels is None \
                else np.concatenate((img_labels, labels_c), axis=0)

        for label in range(5, 10):
            color = colors[label]
            data_c = data[labels == label]  # 某个类别的数据

            index = np.array(color == 'r') + np.array(color == 'b')

            data_c = data_c[index]
            color_c = color[index]
            labels_c = [label] * len(data_c)

            imgs = data_c if imgs is None \
                else np.concatenate((imgs, data_c), axis=0)
            img_colors = color_c if img_colors is None \
                else np.concatenate((img_colors, color_c), axis=0)
            img_labels = labels_c if img_labels is None \
                else np.concatenate((img_labels, labels_c), axis=0)

        cls._imgs2file(imgs, img_colors, img_labels, data_dir)

    @classmethod
    def mnist_marginal2file(cls):
        def f():
            os.makedirs(train_data_dir, exist_ok=True)
            os.makedirs(test_data_dir, exist_ok=True)

            train_imgs, train_colors, train_labels = None, None, None
            test_imgs, test_colors, test_labels = None, None, None
            for label in range(10):
                color_l = colors[label]
                data_l = data[labels == label]  # 某个类别的数据

                for i, c in enumerate(('r', 'y', 'g', 'b')):
                    index = np.array(color_l == c)
                    data_c = data_l[index]  # 某个颜色的数据
                    color_c = color_l[index]
                    labels_c = [label] * len(data_c)

                    split = int(p[i] * len(data_c))  # 分割数据集

                    train_imgs = data_c[:split] if train_imgs is None \
                        else np.concatenate((train_imgs, data_c[:split]), axis=0)
                    train_colors = color_c[:split] if train_colors is None \
                        else np.concatenate((train_colors, color_c[:split]), axis=0)
                    train_labels = labels_c[:split] if train_labels is None \
                        else np.concatenate((train_labels, labels_c[:split]), axis=0)

                    test_imgs = data_c[split:] if test_imgs is None \
                        else np.concatenate((test_imgs, data_c[split:]), axis=0)
                    test_colors = color_c[split:] if test_colors is None \
                        else np.concatenate((test_colors, color_c[split:]), axis=0)
                    test_labels = labels_c[split:] if test_labels is None \
                        else np.concatenate((test_labels, labels_c[split:]), axis=0)

            cls._imgs2file(train_imgs, train_colors, train_labels, train_data_dir)
            cls._imgs2file(test_imgs, test_colors, test_labels, test_data_dir)

        root = './dataset/mnist_diversity'

        data = cls.data.squeeze()
        labels = cls.labels

        colors = []
        for label in range(10):
            len_c = len(labels[labels == label])
            len1_4 = int(len_c / 4)
            color = ['r'] * len1_4 + ['y'] * len1_4 + ['g'] * len1_4 + ['b'] * (len_c - len1_4 * 3)

            random.seed(2)
            random.shuffle(color)
            colors.append(np.array(color))

        # [（0.5, 0.5, 0.5, 0.5)
        p = (0.5, 0.5, 0.5, 0.5)
        train_data_dir, test_data_dir = f'{root}/0/train', f'{root}/0/test'
        f()

        # (0.9, 0.7, 0.3, 0.1)
        p = (0.9, 0.7, 0.3, 0.1)
        train_data_dir, test_data_dir = f'{root}/1/train', f'{root}/1/test'
        f()

        # (1, 0.9, 0.1, 0)
        p = (1, 0.9, 0.1, 0)
        train_data_dir, test_data_dir = f'{root}/2/train', f'{root}/2/test'
        f()

        # (1, 1, 0, 0)
        p = (1, 1, 0, 0)
        train_data_dir, test_data_dir = f'{root}/3/train', f'{root}/3/test'
        f()

    @classmethod
    def mnist_conditional2file(cls):
        def f():
            os.makedirs(train_data_dir, exist_ok=True)
            os.makedirs(test_data_dir, exist_ok=True)

            train_imgs, train_colors, train_labels = None, None, None
            test_imgs, test_colors, test_labels = None, None, None
            for label in range(0, 5):
                color_l = colors[label]
                data_l = data[labels == label]  # 某个类别的数据

                for i, c in enumerate(('r', 'y', 'g', 'b')):
                    index = np.array(color_l == c)
                    data_c = data_l[index]  # 某个颜色的数据
                    color_c = color_l[index]
                    labels_c = [label] * len(data_c)

                    split = int(p[i] * len(data_c))  # 分割数据集

                    train_imgs = data_c[:split] if train_imgs is None \
                        else np.concatenate((train_imgs, data_c[:split]), axis=0)
                    train_colors = color_c[:split] if train_colors is None \
                        else np.concatenate((train_colors, color_c[:split]), axis=0)
                    train_labels = labels_c[:split] if train_labels is None \
                        else np.concatenate((train_labels, labels_c[:split]), axis=0)

                    test_imgs = data_c[split:] if test_imgs is None \
                        else np.concatenate((test_imgs, data_c[split:]), axis=0)
                    test_colors = color_c[split:] if test_colors is None \
                        else np.concatenate((test_colors, color_c[split:]), axis=0)
                    test_labels = labels_c[split:] if test_labels is None \
                        else np.concatenate((test_labels, labels_c[split:]), axis=0)
            for label in range(5, 10):
                color_l = colors[label]
                data_l = data[labels == label]  # 某个类别的数据

                for i, c in enumerate(('r', 'y', 'g', 'b')):
                    index = np.array(color_l == c)
                    data_c = data_l[index]  # 某个颜色的数据
                    color_c = color_l[index]
                    labels_c = [label] * len(data_c)

                    q = 1 - p[i]
                    split = int(q * len(data_c))  # 分割数据集

                    train_imgs = data_c[:split] if train_imgs is None \
                        else np.concatenate((train_imgs, data_c[:split]), axis=0)
                    train_colors = color_c[:split] if train_colors is None \
                        else np.concatenate((train_colors, color_c[:split]), axis=0)
                    train_labels = labels_c[:split] if train_labels is None \
                        else np.concatenate((train_labels, labels_c[:split]), axis=0)

                    test_imgs = data_c[split:] if test_imgs is None \
                        else np.concatenate((test_imgs, data_c[split:]), axis=0)
                    test_colors = color_c[split:] if test_colors is None \
                        else np.concatenate((test_colors, color_c[split:]), axis=0)
                    test_labels = labels_c[split:] if test_labels is None \
                        else np.concatenate((test_labels, labels_c[split:]), axis=0)

            cls._imgs2file(train_imgs, train_colors, train_labels, train_data_dir)
            cls._imgs2file(test_imgs, test_colors, test_labels, test_data_dir)

        root = './dataset/mnist_correlation'

        data = cls.data.squeeze()
        labels = cls.labels

        colors = []
        for label in range(10):
            len_c = len(labels[labels == label])
            len1_4 = int(len_c / 4)
            color = ['r'] * len1_4 + ['y'] * len1_4 + ['g'] * len1_4 + ['b'] * (len_c - len1_4 * 3)

            random.seed(2)
            random.shuffle(color)
            colors.append(np.array(color))

        # [（0.5, 0.5, 0.5, 0.5)
        p = (0.5, 0.5, 0.5, 0.5)
        train_data_dir, test_data_dir = f'{root}/0/train', f'{root}/0/test'
        f()
        p = (0.6, 0.6, 0.4, 0.4)
        train_data_dir, test_data_dir = f'{root}/1/train', f'{root}/1/test'
        f()
        p = (0.7, 0.7, 0.3, 0.3)
        train_data_dir, test_data_dir = f'{root}/2/train', f'{root}/2/test'
        f()
        p = (0.8, 0.8, 0.2, 0.2)
        train_data_dir, test_data_dir = f'{root}/3/train', f'{root}/3/test'
        f()
        p = (1, 1, 0, 0)
        train_data_dir, test_data_dir = f'{root}/4/train', f'{root}/4/test'
        f()


if __name__ == '__main__':
    ColorMNIST.mnist_conditional2file()
