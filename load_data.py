import scipy.io as sio
import numpy as np
import torch


data_names = {"orl": "ORL_32x32",
              "coil20": "COIL20",
              "coil100": "COIL100"}


def get_data(data_name, device):
    filename = "./datasets/" + data_names[data_name] + ".mat"

    # 读取数据
    data = sio.loadmat(filename)
    features, labels = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']

    # 数据简单处理
    features = torch.from_numpy(features).float().to(device)
    labels = np.squeeze(labels - np.min(labels))

    return features, labels


if __name__ == "__main__":
    for name in data_names.keys():
        if name in ["nuswide", "caltech20"]:
            continue

        x, y = get_data(name, 'cpu')
        print(name)
        print(x.shape)
        print(len(np.unique(y)))
