def time_test(repeat_times=5000):
    """
    函数运行效率装饰器
    @see https://blog.csdn.net/Shenpibaipao/article/details/88044951
    :param repeat_times: 重复次数
    :return: 原函数结果
    """
    def func_acceptor(func):
        def wrapper(*arg, **args):
            import time
            start, result = time.time(), {}
            for i in range(repeat_times):  # 重复5000次计算
                result = func(*arg, **args)
            print("function[%s]" % func.__name__, "get result:", result,
                  "while repeat %d times, using time: %.2f(s)" % (repeat_times, time.time()-start))
            return result
        return wrapper
    return func_acceptor


@time_test()
def solution(data):
    """
    基于动态规划解 0-1 背包问题
    :param data: 数据集
    :return: 最优价值和放入背包的物品序号
    """
    total, w, v, answer = data.values()
    item_num = len(w)  # item_num 个物品，至多放 item_num 次
    # dp[第i次尝试放入][此时可用的背包容量j]
    dp = [[0 for _1 in range(total + 1)] for _2 in range(item_num + 1)]  # which is : int dp[item_num+1][total+1] = {0}
    trace = [[[] for _1 in range(total + 1)] for _2 in range(item_num + 1)]  # 放入背包中的物品的最优组合的序号list
    for i in range(item_num):  # 第i+1次尝试放入第i个物品(物品序号从0起||显然，尝试次数从1起,即i+1)
        for j in range(total, -1, -1):  # form $total to $0, step=-1  # 第i+1次尝试放入时的现有背包可用容量
            (trace[i + 1][j], dp[i + 1][j]) = (trace[i][int(j - w[i])] + [i], dp[i][j - w[i]] + v[i]) \
                if j >= w[i] and dp[i][j - w[i]] + v[i] > dp[i][j] else (trace[i][j], dp[i][j])
            # which is:
            # if j >= w[i] and dp[i][j - w[i]] + v[i] > dp[i][j]:  # 第[i+1]次尝试放入时选择放入第[i]个物品
            #     dp[i + 1][j] = dp[i][j - w[i]] + v[i]  # 更新第i+1次尝试放入的推测值
            #     trace[i + 1][j] = trace[i][int(j - w[i])] + [i]
            # else:  # 跳过物品，拷贝"前[i]次尝试放入"的堆栈数据到[i+1]上, 事实上可以直接放弃拷贝过程节省程序开销
            #     dp[i + 1][j] = dp[i][j]
            #     trace[i + 1][j] = trace[i][j]
    return {
        "max_value": dp[item_num][total],
        "package_trace": trace[item_num][total]
    }


@time_test()
def solution_compress(data):
    """
    带空间压缩的 0-1 背包问题
    :param data: 数据集
    :return: 最优价值和放入背包的物品序号
    """
    total, w, v, answer = data.values()
    item_num = len(w)
    dp = [0 for _1 in range(total + 1)]  # which is : int dp[total+1] = {0}
    trace = [[] for _1 in range(total + 1)]
    for i in range(item_num):
        for j in range(total, -1, -1):  # 必需保证是从后往前更新dp[]
            # update trace[] before dp[]
            trace[j], dp[j] = (trace[int(j - w[i])] + [i], dp[j - w[i]] + v[i])\
                if j >= w[i] and dp[j - w[i]] + v[i] > dp[j] else (trace[j], dp[j])
            # which is :
            # if j >= w[i] and dp[j - w[i]] + v[i] > dp[j]:
            #     dp[j] = dp[j - w[i]] + v[i]
            #     trace[j] = trace[int(j - w[i])] + [i]
            # else:
            #     dp[j] = dp[j]  # 显然这里可以直接break以加快算法速度
            #     trace[j] = trace[j]
    return {
        "max_value": dp[total],
        "package_trace": trace[total]
    }


@time_test()
def solution_speedup(data):
    """
    带空间压缩和加速的 0-1 背包问题
    :param data: 数据集
    :return: 最优价值和放入背包的物品序号
    """
    total, w, v, answer = data.values()
    item_num = len(w)
    dp = [0 for _1 in range(total + 1)]
    trace = [[] for _1 in range(total + 1)]
    for i in range(item_num):
        for j in range(total, w[i]-1, -1):  # form $total to $w[i], step=-1
            trace[j], dp[j] = (trace[int(j - w[i])] + [i], dp[j - w[i]] + v[i]) \
                if dp[j - w[i]] + v[i] > dp[j] else (trace[j], dp[j])
            # which is :
            # if j >= w[i] and dp[j - w[i]] + v[i] > dp[j]:
            #     dp[j] = dp[j - w[i]] + v[i]
            #     trace[j] = trace[int(j - w[i])] + [i]
    return {
        "max_value": dp[total],
        "package_trace": trace[total]
    }


@time_test()
def solution_speedup_plus(data):
    """
    带空间压缩和进一步加速的 0-1 背包问题
    :param data: 数据集
    :return: 最优价值和放入背包的物品序号
    """
    total, w, v, answer = data.values()
    item_num = len(w)
    dp = [0 for _1 in range(total + 1)]
    trace = [[] for _1 in range(total + 1)]
    for i in range(item_num):
        lower = max(total-sum(w[i:]), w[i])  # 修正下限，进一步加速
        for j in range(total, lower-1, -1):
            trace[j], dp[j] = (trace[int(j - w[i])] + [i], dp[j - w[i]] + v[i]) \
                if dp[j - w[i]] + v[i] > dp[j] else (trace[j], dp[j])
    return {
        "max_value": dp[total],
        "package_trace": trace[total]
    }


@time_test(repeat_times=1)
def solution_restrict(data, restrict=True):
    """
    带空间压缩和进一步加速的 0-1 背包问题(兼容约束解决方案)
    :param data: 数据集
    :param restrict: 是否进行装满约束
    :return: 最优价值和放入背包的物品序号
    """
    total, w, v, answer = data.values()
    item_num = len(w)
    specific = float("-inf") if restrict else 0  # 如果存在约束，指定所有背包容量为负无穷大
    # 只有初始状态装了0个物品且尚余容量为0的背包能够恰好被装满
    # 因为无论如何不放任何东西就是合法的解，而放了就表示必须将其占满——
    # 即上一个状态(初始时为放了0个物品的状态)背包满了(合法解)，则下一个状态才可能是满的
    # 如果找不到任何可行解，则dp[total]=negative infinite，即非法解
    dp = [specific for _1 in range(total + 1)]
    dp[0] = 0
    trace = [[] for _1 in range(total + 1)]
    for i in range(item_num):
        lower = max(total-sum(w[i:]), w[i])
        for j in range(total, lower-1, -1):
            trace[j], dp[j] = (trace[int(j - w[i])] + [i], dp[j - w[i]] + v[i]) \
                if dp[j - w[i]] + v[i] > dp[j] else (trace[j], dp[j])
    return {
        "max_value": dp[total],
        "package_trace": trace[total]
    }


if __name__ == '__main__':
    datasets = [
        {
            "total": 8,  # total capacity(weight)
            "weight": [2, 3, 4, 5],  # item weight
            "value": [3, 4, 5, 6],  # item value
            "answer": {  # this is just for testing, can be ignored
                "max_value": 10,
                "package_trace": [1, 3]  # index of item
            }
        }, {
            "total": 1000,
            "weight": [200, 600, 100, 180, 300, 450],
            "value": [6, 10, 3, 4, 5, 8],
            "answer": {
                "max_value": 21,
                "package_trace": [0, 2, 3, 5]  # index of item
            }
        }, {
            "total": 1000,  # 注意：该数据样本用于装满约束
            "weight": [200, 600, 100, 200, 300, 450],
            "value": [6, 10, 3, 4, 5, 8],
            "answer": {  # 该答案集用于装满约束
                "max_value": 20,
                "package_trace": [0, 1, 3]  # index of item
            }
        }
    ]
    # run testing
    solution(datasets[1])
    solution_compress(datasets[1])
    solution_speedup(datasets[1])
    solution_speedup_plus(datasets[1])
    solution_restrict(datasets[2])
