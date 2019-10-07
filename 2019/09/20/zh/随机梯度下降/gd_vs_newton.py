import numpy as np
import matplotlib.pyplot as plt
import time
N = 50
M = 2


def prepare(seed, a0=105, b0=10):
    # 产生数据集
    np.random.seed(seed)
    data = np.random.randn(N, 2)
    x = data[:, 0:2].reshape((N, 2))
    y = f(x, a0, b0).reshape((N, 1))  # 由假设参数a0 b0拟合得到真值
    assert x.shape == (N, 2) and y.shape == (N, 1)
    # print('真实值:\n', y)
    # print('特征值:\n', x)
    return y, x


f = lambda x, a, b: a * x[:, 0] ** 2 + np.sin(b * x[:, 1])

cost = lambda y, x, a, b: (0.5 / N) * (y[:, 0] - f(x, a, b)) ** 2


def grad(y, x, a, b):
    return np.array([np.sum((y[:, 0] - f(x, a, b)) / N * (-x[:, 0] ** 2)),
                     np.sum((y[:, 0] - f(x, a, b)) / N * (-x[:, 1] * np.cos(b * x[:, 1])))
                     ]).reshape((M, 1))


def hessian(y, x, a, b):
    h00 = x[:, 0]**4 / N
    h01 = x[:, 0]**2 * x[:, 1] * np.cos(b * x[:, 1])
    h10 = h01
    h11 = x[:, 1]**2 / N * (y[:, 0] * np.sin(b * x[:, 1])
                            -a * x[:, 0]**2 * np.sin(b*b * x[:, 1]
                            +x[:, 1] * np.cos(2 * b * x[:, 1])))
    return np.array([[np.sum(h00), np.sum(h01)], [np.sum(h10), np.sum(h11)]]).reshape((M, M))


def alg(y, x, a, b):
    # 带有成功失败法的梯度下降
    time_start = time.time()
    epsilon = 1e-1
    costs = []
    lambda0 = 0.1
    while True:
        delta_a, delta_b = grad(y, x, a, b)
        if abs(delta_a) < epsilon and abs(delta_b) < epsilon:
            break
        t0_cost = np.sum(cost(y, x, a, b))
        costs.append(t0_cost)
        delta_lambda, lambda_star = lambda0, 0
        while abs(delta_lambda) >= epsilon:
            t1_cost = np.sum(cost(y, x, a-(lambda_star+delta_lambda) * delta_a, b-(lambda_star+delta_lambda) * delta_b))

            if t1_cost > t0_cost:  # 搜索失败
                delta_lambda *= -0.25
            else:
                t0_cost = t1_cost
                lambda_star = delta_lambda+lambda_star
                delta_lambda *= 2

        a -= lambda_star * delta_a
        b -= lambda_star * delta_b
    print("a* = ", a)
    print("b* = ", b)
    print("time usage:", time.time() - time_start)
    return costs, a, b


def alg2(y, x, a, b):
    time_start = time.time()
    # 普通小步长迭代算法
    epsilon = 1e-1
    costs = []
    lambda_star = 0.1
    while True:
        delta_a, delta_b = grad(y, x, a, b)
        if abs(delta_a) < epsilon and abs(delta_b) < epsilon:
            break
        t0_cost = np.sum(cost(y, x, a, b))
        costs.append(t0_cost)
        a -= lambda_star * delta_a
        b -= lambda_star * delta_b
    print("a2* = ", a)
    print("b2* = ", b)
    print("time usage:", time.time() - time_start)
    return costs, a, b


def alg3(y, x, a, b):
    # 二阶导数测试
    time_start = time.time()
    epsilon = 1e-1
    costs = []
    while True:
        g = grad(y, x, a, b)
        h = hessian(y, x, a, b)
        delta_a, delta_b = g
        lambda_star = g.T.dot(g) / g.T.dot(h).dot(g)
        if abs(delta_a) < epsilon and abs(delta_b) < epsilon:
            break
        t0_cost = np.sum(cost(y, x, a, b))
        costs.append(t0_cost)
        a -= lambda_star * delta_a
        b -= lambda_star * delta_b
    print("a3* = ", a)
    print("b3* = ", b)
    print("time usage:", time.time() - time_start)
    return costs, a, b


def alg4(y, x, a, b):
    # 牛顿法
    time_start = time.time()
    epsilon = 1e-1
    costs = []
    while True:
        g = grad(y, x, a, b)
        h = hessian(y, x, a, b)
        delta_a, delta_b = np.linalg.inv(h).dot(g).reshape((M, 1))
        if abs(delta_a) < epsilon and abs(delta_b) < epsilon:
            break
        t0_cost = np.sum(cost(y, x, a, b))
        costs.append(t0_cost)
        a -= delta_a
        b -= delta_b
    print("a4* = ", a)
    print("b4* = ", b)
    print("time usage:", time.time() - time_start)
    return costs, a, b


if __name__ == '__main__':
    y, x = prepare(seed=1355)
    gd_costs, a_star, b_star = alg(y, x, a=-120, b=5)
    gd_costs2, a_star2, b_star2 = alg2(y, x, a=-120, b=5)
    gd_costs3, a_star3, b_star3 = alg3(y, x, a=-120, b=5)
    gd_costs4, a_star4, b_star4 = alg4(y, x, a=-120, b=5)
    ax = plt.figure().add_subplot(1, 1, 1)
    ax.plot(range(len(gd_costs)), gd_costs, '-',
            range(len(gd_costs2)), gd_costs2, '--',
            range(len(gd_costs3)), gd_costs3, '-.',
            range(len(gd_costs3)), gd_costs3, '-x')
    plt.legend(('Gradient Descent with method of S-F',
                'Gradient Descent',
                'Gradient Descent with SDT',
                'Newton'), loc='upper right')
    ax.set(xlabel='times', ylabel='cost', title='Gradient Descent')
    plt.show()

