1. 首先运行read_split_data.py 将原始数据进行预处理
运行 train_helper.py , 其中有两个参数需要注意，use_cuda（第11行） ：是否使用GPU加速，默认True开启（否则会比较慢）；
  预处理会将数据分为8-fold，f（第18行）控制使用相应的训练、测试数据。
