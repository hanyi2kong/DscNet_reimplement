from configs.cfg_orl import ConfigOrl
from configs.cfg_coil20 import ConfigCoil20
from configs.cfg_coil100 import ConfigCoil100

config_list = {"orl": ConfigOrl,
               "coil20": ConfigCoil20,
               "coil100": ConfigCoil100}


def get_config(name):
    assert name in config_list.keys()

    print("get configs", name)
    return config_list[name]


if __name__ == "__main__":
    cfg = get_config("orl")
    print(cfg)
