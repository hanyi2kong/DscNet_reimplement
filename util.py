import os
import torch


def check_path():
    if os.path.exists("results") is False:
        os.makedirs('results')


def load_model(model, name, state):
    state_dict = torch.load(state + '/%s.pkl' % name)
    model.load_state_dict(state_dict)


def save_mode(model, name):
    torch.save(model.state_dict(), 'results/%s-model.pkl' % name)
