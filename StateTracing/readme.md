1. 首先运行read_split_data.py 将原始数据进行预处理。
2. 运行 train_helper.py , 其中有两个参数需要注意，use_cuda（第11行） ：是否使用GPU加速，默认True开启（否则会比较慢）；
  预处理会将数据分为8-fold，f（第18行）控制使用相应的训练、测试数据。
3. init.dat 和 items.dat 分别记录每道题的初始状态，以及item-to-id信息。init.dat需要手动添加；items.dat是read_split_data处理而来，无需手动添加。
4. 在test_helper.py 中可以根据已有的行为流数据对下一个行为进行预测
5. 模型预测正确率约为93%
