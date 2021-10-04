#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/9/27 9:32
# DESCRIPTION:
import torch
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from sklearn.cluster import KMeans
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
from PIL import Image


class ClusteredMNIST:
    mnist = datasets.MNIST(root='../dataset', transform=transforms.ToTensor())
    loader = DataLoader(mnist, batch_size=len(mnist), num_workers=5)
    encoder = tv.models.resnet50(pretrained=True)

    num_class = 10
    num_cluster = 5

    @classmethod
    def t(cls):
        """
        For test
        :return:
        """
        a = cls._get_coding(0)

    @classmethod
    def _get_data_index_list(cls, data_class):
        """
        获取某个数字的index列表
        :param _class:
        :return:
        """
        index_lists = [[] for _ in range(cls.num_class)]

        _, label = next(cls.loader.__iter__())
        label = label.squeeze()
        for idx, label in enumerate(label.tolist()):
            index_lists[label].append(idx)
        return index_lists[data_class]

    @classmethod
    def _get_coding(cls, data_class):
        def _hook(module, input, output):
            coding.append(input[0].cpu().detach())

        index_list = cls._get_data_index_list(data_class)

        data, label = next(cls.loader.__iter__())
        data, label = data.squeeze()[index_list], label[index_list]  # 取出某个数字的数据

        data = torch.stack((data, data, data), dim=-1).squeeze()  # grad -> rgb
        data = data.permute((0, 3, 1, 2))

        coding = []  # 用于接收钩子函数的输出
        cls.encoder.fc.register_forward_hook(_hook)
        cls.encoder(data)

        return coding[0]

    @classmethod
    def coding2pkl(cls):
        coding_list = []
        for data_class in range(cls.num_class):
            coding = cls._get_coding(data_class)
            coding_list.append(coding)
        pickle.dump(coding_list, open('./count/coding_list.pkl', 'wb'))

    @classmethod
    def kmeans2pkl(cls):
        coding_list = pickle.load(open('./count/coding_list.pkl', 'rb'))
        cluster_list = []
        for data_class in tqdm(range(cls.num_class)):
            coding = coding_list[data_class]
            cluster = KMeans(n_clusters=cls.num_cluster, random_state=2).fit(coding)
            cluster_list.append(cluster)
        pickle.dump(cluster_list, open('./count/cluster_list.pkl', 'wb'))

    @classmethod
    def _get_cluster_index_list(cls, data_class):
        """
        获取某个类别聚类后的index列表
        :param _class:
        :return:
        """
        cluster_list = pickle.load(open('./count/cluster_list.pkl', 'rb'))
        cluster = cluster_list[data_class]

        index_list = [[]for _ in range(cls.num_cluster)]
        for i, v in enumerate(cluster.labels_):
            index_list[v].append(i)
        return index_list

    @classmethod
    def _get_ni_info(cls, data_class):
        def _ni_index(cluster_0, cluster_1):
            cluster_0 = cluster_0.numpy()
            cluster_1 = cluster_1.numpy()
            mean_0 = np.mean(cluster_0, axis=0)
            mean_1 = np.mean(cluster_1, axis=0)
            std = np.std(np.concatenate((cluster_0, cluster_1), axis=0))
            z = (mean_0 - mean_1) / std
            ni = np.linalg.norm(z)
            return ni

        coding_list = pickle.load(open('./count/coding_list.pkl', 'rb'))
        coding = coding_list[data_class]

        cluster_index_list = cls._get_cluster_index_list(data_class)

        coding_basic_cluster = coding[cluster_index_list[0]]

        ni_list = []
        for cluster_index in cluster_index_list:
            coding_compared_cluster = coding[cluster_index]
            ni = _ni_index(coding_basic_cluster, coding_compared_cluster)
            ni_list.append(ni)

        ni_rank = sorted(range(len(ni_list)), key=lambda k: ni_list[k])
        ni_list = sorted(ni_list)
        return ni_list, ni_rank

    @classmethod
    def print_ni_info(cls):
        ni_info_list = []
        for data_class in range(cls.num_class):
            ni_info = cls._get_ni_info(data_class)
            print(ni_info)
            ni_info_list.append(ni_info)

        ni, _ = zip(*ni_info_list)
        ni = np.array(ni)
        print(np.mean(ni, axis=0).tolist())  # [ 0.0, 9.22905, 11.084345, 12.117267, 14.091411]

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
        for num_cluster in tqdm(range(cls.num_cluster)):
            data_dir = f'{root}/{str(num_cluster)}'
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)

            count = 0
            for data_class in range(cls.num_class):
                # 获取某个数字类别所有的在盖类别内的聚类index
                cluster_index_list = cls._get_cluster_index_list(data_class)
                # 得到聚类的NI值的排名
                _, rank = cls._get_ni_info(data_class)
                cluster_index = cluster_index_list[rank[num_cluster]]
                # 该聚类对应的data index
                data_index = np.array(cls._get_data_index_list(data_class))[cluster_index].tolist()

                data, label = next(cls.loader.__iter__())
                data, label = data.squeeze()[data_index], label[data_index]
                data = torch.stack((data, data, data), dim=-1)
                data = data.permute((0, 3, 1, 2))

                for img, label in zip(data, label):
                    img = transforms.ToPILImage()(img)
                    name = f'{str(int(label))}_{_num2str(count + 1)}.jpg'
                    img.save(os.path.join(root, f'{str(num_cluster)}/{name}'))
                    count += 1
                print(count)


class ColoredMNIST:
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
        assert c in ['red', 'r', 'yellow', 'y', 'green', 'g', 'cyan', 'c', 'blue', 'b']

        dtype = img.dtype
        h, w = img.shape
        arr = np.reshape(img, [h, w, 1])
        if c == 'red' or c == 'r':
            arr = np.concatenate([arr,
                                  np.zeros((h, w, 1), dtype=dtype),
                                  np.zeros((h, w, 1), dtype=dtype)], axis=2)
        elif c == 'yellow' or c == 'y':
            arr = np.concatenate([arr,
                                  arr,
                                  np.zeros((h, w, 1), dtype=dtype)], axis=2)
        elif c == 'green' or c == 'g':
            arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                                  arr,
                                  np.zeros((h, w, 1), dtype=dtype)], axis=2)

        elif c == 'cyan' or c == 'c':
            arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                                  arr,
                                  arr], axis=2)

        elif c == 'blue' or c == 'b':
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
    def mnist_diversity2file(cls):
        np.random.seed(2)

        root = './dataset/mnist_diversity'

        for i in range(5):
            data_dir = f'{root}/{str(i)}'
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)

            start = 12000 * i
            data = cls.data[start: start + 12000].squeeze()
            labels = cls.labels[start: start + 12000]

            for idx, (img, label) in tqdm(enumerate(zip(data, labels))):
                color = 'red'

                p = np.random.uniform()

                if i == 0:
                    pass
                elif i == 1:
                    if 1/2 <= p:
                        color = 'green'
                elif i == 2:
                    if 1/3 <= p < 2/3:
                        color = 'green'
                    elif 2/3 <= p:
                        color = 'yellow'
                elif i == 3:
                    if 1/4 <= p < 2/4:
                        color = 'green'
                    elif 2/4 <= p < 3/4:
                        color = 'yellow'
                    elif 3/4 <= p:
                        color = 'cyan'
                else:
                    if 1/5 <= p < 2/5:
                        color = 'green'
                    elif 2/5 <= p < 3/5:
                        color = 'yellow'
                    elif 3/5 <= p < 4/5:
                        color = 'cyan'
                    elif 4/5 <= p:
                        color = 'blue'

                img = cls._color_img(img.squeeze().numpy(), color)
                img = img.transpose((2, 0, 1))
                img = torch.from_numpy(img)
                img = transforms.ToPILImage()(img)

                name = f'{str(int(label))}_{cls._num2str(idx + 1)}.jpg'
                img.save(f'{data_dir}/{name}')

    @classmethod
    def mnist_correlation2file(cls):
        np.random.seed(2)

        root = './dataset/mnist_correlation'

        for i in range(5):
            data_dir = f'{root}/{str(i)}'
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)

            start = 12000 * i
            data = cls.data[start: start + 12000].squeeze()
            labels = cls.labels[start: start + 12000]

            for idx, (img, label) in tqdm(enumerate(zip(data, labels))):
                color = 'red' if int(label) < 5 else 'green'
                color_opp = 'green' if int(label) < 5 else 'red'

                if i == 0:
                    # 20% in the first training environment
                    if np.random.uniform() < 0.2:
                        color = color_opp
                elif i == 1:
                    if np.random.uniform() < 0.4:
                        color = color_opp
                elif i == 2:
                    if np.random.uniform() < 0.6:
                        color = color_opp
                elif i == 3:
                    if np.random.uniform() < 0.8:
                        color = color_opp
                else:
                    if np.random.uniform() < 1:
                        color = color_opp

                img = cls._color_img(img.squeeze().numpy(), color)
                img = img.transpose((2, 0, 1))
                img = torch.from_numpy(img)
                img = transforms.ToPILImage()(img)

                name = f'{str(int(label))}_{cls._num2str(idx + 1)}.jpg'
                img.save(f'{data_dir}/{name}')


ClusteredMNIST.t()