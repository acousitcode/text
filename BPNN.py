import numpy as np  # 引入numpy模块
import matplotlib.pyplot as plt  # 引入matplotlib模块


# 生成带有噪声的正弦数据
def generate_data(num_samples):
    x = np.linspace(0, 2 * np.pi, num_samples)
    # 白噪声符合正态分布，并使用了0.1来控制噪声的幅度
    y = np.sin(x) + 0.1 * np.random.randn(num_samples)
    # print('y=',y)
    # print('noise=',np.random.randn(num_samples))
    # 将x和y都输出为num_samples行一列的数组
    return x.reshape(-1, 1), y.reshape(-1, 1)


# 定义激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# 初始化网络参数
def initialize_parameters(input_size, hidden_size, output_size):
    # 输入的基本参数包括输入层、隐藏层和输出层大大小，实际对应各层的行数
    # W1是从输入层进入隐藏层的连接权重
    # W1=先按照正态分布的形式生成随机数，然后乘以系数0.01
    # W1是一个input_size行hidden_size列的矩阵
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    # b1是一个1行hidden_size列的矩阵
    b1 = np.zeros((1, hidden_size))
    # W2是从输入层进入隐藏层的连接权重
    # W2=先按照正态分布的形式生成随机数，然后乘以系数0.01
    # W2是一个hidden_size行output_size列的矩阵
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    # b2是一个1行output_size列的矩阵
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2


# 前向传播
def forward_propagation(X, W1, b1, W2, b2):
    # 按照矩阵乘法的运算原则，将X和W1相乘，然后叠加偏置量b1
    # X、W1和b1都是矩阵数据，会按照顺序注意叠加，类似于x[1]w1[1]+b1[1]
    # 这一步的计算是从输入层接入隐藏层
    # Z1是一个num_samples行hidden_size列的矩阵
    Z1 = np.dot(X, W1) + b1
    # 把A1代入激活函数
    # 这一步的计算式获得隐藏层的输出
    # 获得的A1是一个num_samples行hidden_size列的矩阵向量
    A1 = sigmoid(Z1)
    # 按照矩阵乘法的运算原则，将A1和W2相乘，然后叠加偏置量b2
    # A1、W2和b2都是矩阵数据，会按照顺序注意叠加，类似于A1[1]w2[1]+b2[1]
    # 获得的Z2是一个num_samples行output_size列的矩阵
    Z2 = np.dot(A1, W2) + b2
    # 这一步的计算是从隐藏层接入输出层，可以理解为所有权重都是1，偏置量都是0
    # 获得的A2是一个num_samples行output_size列的矩阵
    A2 = Z2
    return Z1, A1, Z2, A2


# 计算损失（均方误差）
def compute_cost(A2, Y):
    # 将矩阵Y的行数取出来赋值给m
    m = Y.shape[0]
    # A2是输出值，先将所有输出值和目标值作差，然后再对差进行平方求和，再对差平方和取均值
    cost = (1 / (2 * m)) * np.sum(np.square(A2 - Y))
    return cost


# 反向传播
def backward_propagation(X, Y, Z1, A1, A2, W1, W2):
    # 将矩阵X的行数取出来赋值给m
    m = X.shape[0]
    # A2是输出值，将所有输出值和目标值作差后赋值给DZ2
    # dZ2是一个num_samples行output_size列的矩阵
    dZ2 = A2 - Y
    # 首先将A1转置为hidden_size行num_samples列的矩阵A1.T
    # 然后A1和dZ2按照矩阵乘法的原则进行计算获得dW2
    # dW2是一个hidden_size行output_size列的矩阵
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    # db2是将所有的dZ2叠加后求平均值
    # axis=0表示每列从上到下求和
    # dZ2有output_size列，所以会获得output_size个值
    # keepdims=True会将output_size个值记录为1行output_size列的矩阵
    # db2表示将1行output_size列的矩阵取均值，依然是1行output_size列的矩阵
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
    # 首先将W2转置为output_size行hidden_size列的矩阵
    # dZ2是一个num_samples行output_size列的矩阵
    # dZ2和W2.T按照矩阵乘法的原则计算后，获得num_samples行hidden_size列的矩阵
    # Z1是一个num_samples行hidden_size列的矩阵
    # sigmoid_derivative(Z1)是把Z1代入激活函数的导函数的计算值，是一个num_samples行hidden_size列的矩阵
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)
    # dZ1是np.dot(dZ2, W2.T)和sigmoid_derivative(Z1)相同位置的元素彼此相乘获得的
    # dZ1是一个num_samples行hidden_size列的矩阵
    # 首先将X转置成一个一行num_samples列的矩阵X.T
    # 然后X.T再和num_samples行hidden_size列的矩阵dZ1相乘，获得dW1
    # dW1是一个num_samples行hidden_size列的矩阵
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    # dZ1先每一列求和，获得hidden_size个数，keepdims=True将求和结果转化为1行hidden_size列矩阵，然后再求均值
    # db1是一个一行hidden_size列的矩阵
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2


# 更新参数
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    # W1是一个input_size行hidden_size列的矩阵
    # dW1是一个num_samples行hidden_size列的矩阵
    # 实际上输入的input_size行=num_samples行
    # 新获得的W1也是num_samples行hidden_size列的矩阵
    W1 = W1 - learning_rate * dW1
    # b1是一个1行hidden_size列的矩阵
    # db1是一个一行hidden_size列的矩阵
    # 新获得的b1是一个一行hidden_size列的矩阵
    b1 = b1 - learning_rate * db1
    # W2是一个hidden_size行output_size列的矩阵
    # dW2是一个hidden_size行output_size列的矩阵
    # 新获得的W2是一个hidden_size行output_size列的矩阵
    W2 = W2 - learning_rate * dW2
    # b2是一个1行output_size列的矩阵
    # db2是一个1行output_size列的矩阵
    # 新获得的b2是一个一行output_size列的矩阵
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2


# 训练模型
def train_model(X, Y, hidden_size, num_iterations, learning_rate):
    # 定义input_size是X的列数
    input_size = X.shape[1]
    # 定义output_size是Y的列数
    output_size = Y.shape[1]
    # 调用initialize_parameters()函数获得连接权重和偏置量
    # W1是按照正态分布获得的随机数值
    # b1是纯0矩阵
    # W2是按照正态分布获得的随机数值
    # b2是纯0矩阵
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    # 定义空矩阵costs
    costs = []

    # 迭代运算
    for i in range(num_iterations):
        # 调用forward_propagation()函数获得正向运算值
        # Z1是将输入层连接隐藏层的权重代入X，再叠加偏置量后的输出
        # A1是将Z1代入激活函数后的输出
        # Z2是将隐藏层连接输出层的权重代入A1，再叠加偏置量后的输出
        # A2是Z2的直接赋值，没有新的变化
        # Z1是个线性变化量，A2叠加了激活函数中的非线性因素
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        # 运算值和目标值差分
        cost = compute_cost(A2, Y)
        # 计算连接权重和偏置量的逆向变化量
        # dW2是A1的转置和(A2-Y)求均值后再矩阵相乘的值
        # db2是(A2-Y)按列求和之后再求均值后的值
        # dW1是X的转置和dZ1矩阵相乘的值
        # dZ1是(A2-Y)和W2的转置先矩阵相乘，再乘以Z1代入激活函数导函数后的值
        # db1是dZ1列求和之后再求均值后的值
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, A2, W1, W2)
        # 更新所有连接权重和偏置量
        # W1是将上一步的dW1和学习效率相乘后，和原W1作差后的值
        # b1是将上一步的db1和学习效率相乘后，和原b1作差后的值
        # W2是将上一步的dW2和学习效率相乘后，和原W2作差后的值
        # b2是将上一步的db2和学习效率相乘后，和原b2作差后的值
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")

    return W1, b1, W2, b2, costs


# 主函数
def main():
    # 生成数据
    num_samples = 100
    X, Y = generate_data(num_samples)

    # 训练模型
    hidden_size = 10
    num_iterations = 1000
    learning_rate = 0.1
    # 训练参数
    W1, b1, W2, b2, costs = train_model(X, Y, hidden_size, num_iterations, learning_rate)

    # 生成测试数据
    x_test = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
    y_test = np.sin(x_test)

    # 进行预测
    # 预测只在往前传递的运算中执行
    _, _, _, y_pred = forward_propagation(x_test, W1, b1, W2, b2)

    # 绘制结果
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X, Y, label='Training Data', color='blue')
    plt.plot(x_test, y_test, label='True Function', color='green')
    plt.plot(x_test, y_pred, label='Predicted Function', color='red')
    plt.legend()
    plt.title('BP Neural Network')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, num_iterations, 100), costs)
    plt.title('Cost over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')

    plt.show()


if __name__ == "__main__":
    main()