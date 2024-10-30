这个库里存放着文章“A new Air-sea flux algorithm based on deep learning and its application in Earth System Model”的代码和数据。

# 第一部分  python 训练模型代码 lhfDPL.py
个程序是一个TBF的实例程序的torch部分，

第一步：需要输入训练数据，验证数据，测试数据，
        1.计算一个y所需要的x1，x2...xm,这个是m = feature_count .

第二步：网络架构设计：设置有几层，每层有几个节点，哪类激活函数
        1.每层有几个节点在point_count,feature,n = point_count 设置；
        2.有几层这个在nn.sequential{}中设置，然后在后面的o = {激活函数的个数} 
        3.在nn.sequential 中激活函数和线性层交替排列，现代深度学习一般relu为激活函数          也可以使sigmod、tanh等，这个在torch最好设为一样的，在KBF目前只能设置成一样的。

第三步：设置优化方法，主要是对优化方法，npoch,batch_size,等进行修改，以达到最好的优化效果，得到最优参数W,b,c,d

第四步：查看训练的结果，两张图，方差，均方根误差等决定需不需要再次重复上面步骤。

第五步：在训练效果很好，得到正确参数的情况下，传输模型参数先输出为txt文件，
        然后用fortran读取到KBF中。
        1.检查m,n,o,function_kind,shuchucanshu.txt
        2.检查w_input.txt w_dense.txt w_output.txt
        3.检查b_input.txt b_dense.txt b_output.txt 

第六步：将*.txt 传输到torch_bridge_fortran 文件夹中
之后的操作留给fortran程序来解决。


# 第二部分  Fortran 模型在模式计算部分 TBF中代码
