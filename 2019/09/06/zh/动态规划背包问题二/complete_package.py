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
            print("function[%s]" % func.__name__, result,
                  "while repeat %d times, using time: %.2f(s)" % (repeat_times, time.time() - start))
            return result

        return wrapper

    return func_acceptor


def complete_package(encoder_type='full'):
    """
    全量背包在 0-1 背包上的装饰器
    :param encoder_type: 编码器类型 full | compress
    :return: 全量背包的答案解集
    """
    def func_acceptor(func):
        def wrapper(*arg, **args):
            import math
            total, w_o, v_o, answer = arg[0].values()  # data = arg[0]
            w, v, item_num, size_counter = [], [], len(w_o), lambda x: 1
            compress_mode_on = encoder_type == 'compress'

            if not compress_mode_on == 'full':
                size_counter = lambda x: max(total // x, 1)  # 至少为1件
            else:
                size_counter = lambda x: max(math.floor(math.log(total / x, 2)) + 1, 1)

            w, v, mask = encoder(total, w_o, v_o, size_counter, compress_mode_on)  # encode
            # print("w, v encoder to", w, v)

            arg[0]["weight"], arg[0]["value"] = w, v  # 传入编码后的物品重量和价值列表
            result = func(*arg, **args)  # 调用0-1背包的解决方案
            arg[0]["weight"], arg[0]["value"] = w_o, v_o  # 还原修改过后的数据
            # print("origin package_trace =", result["package_trace"])

            # rebuild trace
            # 如果不需要入包的物品的信息可以跳过解码以加快算法运算速率
            result["package_trace"] = decoder(w_o, result["package_trace"], size_counter, mask, compress_mode_on, w)
            return result

        return wrapper

    return func_acceptor


def encoder(total, w_o, v_o, size_counter, compress_mode_on=True):
    """
    采用全量编码的方式重组物品序列
    :param total: 背包容量上限
    :param w_o: 原物品序列重量列表
    :param v_o: 原物品序列价值列表
    :param size_counter: 编码长度计算器
    :param compress_mode_on: logN 编码模式
    :return: 编码后的物品序列价值列表
    """
    item_num, w, v, mask = len(w_o), [], [], get_mask(total, w_o, v_o)

    for i in range(item_num):
        if mask[i] == 1:
            continue
        size = size_counter(w_o[i])
        if not compress_mode_on:  # 全量输出模式 : 添加 ⌊total/v[i]⌋ 件相同的物品
            w += [w_o[i] for _ in range(size)]
            v += [v_o[i] for _ in range(size)]
        else:  # 压缩模式 : 添加 ⌊log(total/v[i])⌋ 件相同的物品
            w += [w_o[i] * (2 ** _) for _ in range(size)]
            v += [v_o[i] * (2 ** _) for _ in range(size)]
    return w, v, mask


def decoder(w_o, trace_o, size_counter, mask, compress_mode_on=True, w=None):
    """
    采用全量编码的方式重组物品序列
    :param w_o: 原物品序列重量列表
    :param trace_o: 原物品序列选择列表
    :param size_counter: 编码长度计算器
    :param mask: 过滤掩码，当其等于 1 时直接进行跳过该物品
    :param compress_mode_on: logN 编码模式
    :param w: 编码后的物品序列列表，用于解码，当且仅当compress_mode_on=True 启用
    :return: 编码后的物品选择列表
    """
    item_num, trace, start = len(w_o), [], 0
    for i in range(item_num):
        if mask[i] == 1:
            continue
        size = size_counter(w_o[i])  # 为了trace可追踪，至少为1件
        if not compress_mode_on:
            num = len(list(filter(lambda x: start <= x < start + size, trace_o)))
        else:
            num, codes = 0, list(filter(lambda x: start <= x < start + size, trace_o))
            assert compress_mode_on and w is not None
            for v in codes:
                num += w[v] // w_o[i]
        if num > 0:
            trace += [(i, num)]
        start += size
    return trace


def get_mask(total, w, v):
    """
    得到掩码
    :param total: 背包总容量
    :param w: 原物品序列重量列表
    :param v: 原物品序列价值列表
    :return:
    """
    item_num = len(w)
    mask = [0 for _1 in range(item_num + 1)]
    for i in range(item_num):  # 产生掩码 复杂度为 O(item_num^2)
        if w[i] > total:
            mask[i] = 1
            continue
        for j in range(i + 1, item_num, 1):
            if mask[i] == 1:
                break
            a, b = w[i] <= w[j], v[i] >= v[j]
            c, d = w[i] >= w[j], v[i] <= v[j]
            picked = j if a and b else (i if c and d else -1)
            if picked > 0:
                mask[picked] = 1
                break
    return mask


@time_test(repeat_times=5000)
@complete_package(encoder_type='full')  # 打上该装饰器，通过编码原完全背包问题为0-1背包问题
def solution_full(data, restrict=True):
    return zero_one_solution(data, restrict)


@time_test(repeat_times=5000)
@complete_package(encoder_type='compress')  # 打上该装饰器，通过编码原完全背包问题为0-1背包问题
def solution_compress(data, restrict=True):
    return zero_one_solution(data, restrict)


def zero_one_solution(data, restrict=True):
    """
    0-1 背包问题(兼容约束解决方案)
    空间复杂度 O(total)，时间复杂度 O(total*item_num')，item_num 经过修正加速
    :param data: 数据集
    :param restrict: 是否进行装满约束 @see https://blog.csdn.net/Shenpibaipao/article/details/90961776#_182
    :return: 最优价值和放入背包的物品序号
    """
    total, w, v, answer = data.values()
    item_num = len(w)
    specific = float("-inf") if restrict else 0  # 兼容满背包约束方案
    dp = [specific for _1 in range(total + 1)]
    dp[0] = 0
    trace = [[] for _1 in range(total + 1)]

    for i in range(item_num):  # 采用01背包的方式解决
        lower = max(total - sum(w[i:]), w[i])
        for j in range(total, lower - 1, -1):
            trace[j], dp[j] = (trace[int(j - w[i])] + [i], dp[j - w[i]] + v[i]) \
                if dp[j - w[i]] + v[i] > dp[j] else (trace[j], dp[j])

    return {
        "max_value": dp[total],
        "package_trace": trace[total]
    }


@time_test(repeat_times=5000)
def solution_speedup(data, restrict=True):
    """
    完全背包问题(兼容约束解决方案)
    空间复杂度 O(total)，时间复杂度 O(total*item_num)
    :param data: 数据集
    :param restrict: 是否进行装满约束 @see https://blog.csdn.net/Shenpibaipao/article/details/90961776#_182
    :return: 最优价值和放入背包的物品序号
    """
    total, w, v, answer = data.values()
    item_num, mask = len(w), get_mask(total, w, v)
    specific = float("-inf") if restrict else 0  # 兼容满背包约束方案
    dp = [specific for _1 in range(total + 1)]
    dp[0] = 0
    trace = [[] for _1 in range(total + 1)]

    for i in range(item_num):
        if mask[i] == 1:
            continue
        for j in range(w[i], total+1, 1):  # 修改此处以实现完全背包 -> for(i=w[i];i<=total;i++)
            trace[j], dp[j] = (trace[int(j - w[i])] + [i], dp[j - w[i]] + v[i]) \
                if dp[j - w[i]] + v[i] > dp[j] else (trace[j], dp[j])

    # 重新编码轨迹，如果不需要输出轨迹或重编译轨迹可直接注释以跳过
    temp_trace, trace[total] = trace[total], []
    for i in range(len(temp_trace)):
        v = temp_trace[i]
        if i > 0 and temp_trace[i-1] == v:  # 依赖逻辑短路保证数组不越界
            continue
        trace[total] += [(v, temp_trace.count(v))]

    return {
        "max_value": dp[total],
        "package_trace": trace[total]
    }


if __name__ == '__main__':
    datasets = [
        {
            "total": 8,  # total capacity(weight)
            "weight": [2, 3, 4, 5],  # item weight
            "value": [2, 4, 5, 6],  # item value
            "answer": {  # this is just for testing, can be ignored
                "max_value": 10,
                "package_trace": [(0, 1), (1, 2)]  # index of item, (i, j) means take item[0] j times
            }
        }, {
            "total": 1000,
            "weight": [200, 600, 100, 180, 300, 450],
            "value": [6, 10, 3, 4, 5, 8],
            "answer": {
                "max_value": 30,
                "package_trace": [(0, 5)]  # index of item
            }
        }, {
            "total": 1000,  # 注意：该数据样本用于装满约束
            "weight": [201, 600, 100, 200, 300, 450],
            "value": [6, 6, 1, 1, 5, 8],
            "answer": {  # 该答案集用于装满约束
                "max_value": 17,
                "package_trace": [(2, 1), (5, 2)]  # index of item
            }
        }, {
            "total": 1000,  # 注意：该数据样本用于测试mask过滤器
            "weight": [199, 600, 100, 200, 300, 450],
            "value": [6, 6, 1, 1, 5, 8],
            "answer": {  # 该答案集用于装满约束
                "max_value": 17,
                "package_trace": [(2, 1), (5, 2)]  # index of item
            }
        }
    ]
    # run testing
    solution_full(datasets[1], restrict=False)
    solution_compress(datasets[1], restrict=False)
    solution_compress(datasets[0], restrict=True)
    solution_speedup(datasets[2], restrict=True)
