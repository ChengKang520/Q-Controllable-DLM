# coding='utf-8'
"""t-SNE 对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE


def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data[:100]
    label = digits.target[:100]
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    plot_emd = plt

    plot_emd.figure(figsize=(6, 5))
    for i in range(data.shape[0]):
        plot_emd.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plot_emd.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plot_emd.xticks([])
    plot_emd.yticks([])

    plot_emd.show()
    plot_emd.savefig('tSNE_embedding.png')
    plot_emd.close()


def main():
    data, label, n_samples, n_features = get_data()

    print('#######################   data!   #######################')
    print(data.shape)
    print(data)
    print('#######################   data!   #######################')

    print('#######################   label!   #######################')
    print(label.shape)
    print(label)
    print('#######################   label!   #######################')


    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)


    result = tsne.fit_transform(data)

    print('#######################   Drawing!   #######################')
    title = 't-SNE embedding'
    plot_embedding(result, label, title)


if __name__ == '__main__':
    main()
