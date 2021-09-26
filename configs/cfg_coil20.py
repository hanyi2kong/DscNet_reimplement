class ConfigCoil20:
    # 数据集参数
    num_cluster = 20
    num_sample = 1440

    # 网络参数
    channels = [1, 15]
    kernels = [3]

    # 训练参数
    epochs = 40
    lr = 0.001
    weight_coe = 1.0
    weight_self_exp = 75

    # post clustering parameters
    alpha = 0.04  # threshold of C
    dim_subspace = 12  # dimension of each subspace
    ro = 8  #

    # 其它参数
    comment64 = False
    show_freq = 1
