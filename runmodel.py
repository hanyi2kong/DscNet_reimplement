import torch
from torch import optim
import logging

from config import get_config
from load_data import get_data
from network import DSCNet
from util import load_model, save_mode
from post_clustering import spectral_clustering
from evaluate import get_score


class RunModel:
    def __init__(self, name):
        # get configs
        self.cfg = get_config(name)
        # get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # get dataloader
        self.features, self.labels = get_data(name, self.device)
        # set name
        self.name = name

    def get_param(self):
        cfg = self.cfg
        return cfg.epochs, cfg.weight_coe, cfg.weight_self_exp, cfg.num_cluster, cfg.dim_subspace, cfg.alpha, cfg.ro, cfg.comment64

    def train_dsc(self):
        # 模型参数
        cfg = self.cfg
        model = DSCNet(num_sample=cfg.num_sample, channels=cfg.channels, kernels=cfg.kernels).to(self.device)
        load_model(model.ae, self.name, 'pretrained_weights_original')

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        epochs, weight_coe, weight_self_exp, num_cluster, dim_subspace, alpha, ro, comment64 = self.get_param()
        x = self.features
        y = self.labels
        for epoch in range(epochs):
            x_recon, z, z_recon = model(x)
            loss = model.loss_fn(x, x_recon, z, z_recon, weight_coe=weight_coe, weight_self_exp=weight_self_exp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch % 1 == 0 or epoch == epochs - 1) and epoch >= 0:
                coe = model.self_expression.Coefficient.detach().to('cpu').numpy()
                y_pred = spectral_clustering(coe, num_cluster, dim_subspace, alpha, ro, comment64)
                acc, nmi = get_score(y, y_pred)
                print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' % (epoch, loss.item() / y_pred.shape[0], acc, nmi))

        save_mode(model, self.name)


if __name__ == "__main__":
    print("a")
    a = RunModel("orl")
    print(len(a.features))
    print(a.features[0].shape, a.features[1].shape, a.features[0].dtype)
