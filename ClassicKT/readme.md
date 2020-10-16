1. 首先运行 preprocess_classic_KT.py 改其中原始数据的路径，将原始数据处理成可以识别的文件。
2. 然后运行 trainner.py 
3. 模型训练过程默认开 cuda加速,如果计算机不支持，将trainner.py 第13行 use_cuda = True 改成 use_cuda = False
4. 模型正确率约为77%，auc约为0.83~0.85.
5. 这里的DKT与经典的DKT实现稍有不同，模型更为简化，对数据的处理（尤其是需要预测的target）更为方便。
