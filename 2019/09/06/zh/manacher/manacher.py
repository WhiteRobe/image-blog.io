def manacher(st, placeholder="#"):
    # 去除奇偶影响
    m_str = placeholder+placeholder.join(st)+placeholder
    # 初始化回文序列 p=[0 1 0 0 ...]
    p = [0] * len(m_str)
    p[1] = 1
    # center初始化为1
    c = 1
    m_str_len = len(m_str)
    for i in range(2, m_str_len):
        # mirror 与 i 关于 center 对称
        m = 2 * c - i

        # 判断p[m]与p[i]是否一致(m的左侧覆盖超出了c的左侧覆盖)
        # 若不一致则进行扩展
        # 注意：python中负索引是合理索引值，其它语言应自行考虑m的越界情况
        if m < 0 or m-p[m] <= c-p[c]:
            # print(i, m, c, m - p[m], c - p[c], "扩展", p[i])
            # 最大扩展半径
            r = min(i, m_str_len-i-1)
            # 最大已知回文半径
            p[i] = max(0, c + p[c] - i)
            for j in range(p[i], r, 1):
                # print(p[i], r, i, j, i - j - 1, i + j + 1, m_str_len)
                if m_str[i - j - 1] == m_str[i + j + 1]:
                    p[i] += 1
                else:
                    break
        else:
            # 根据对称性质复制回文序列
            p[i] = p[m]

        # 更新最大回文串的索引值
        # 取最先找到的一个
        if p[i] > p[c]:
            c = i

    return m_str[c-p[c]:c+p[c]+1].replace(placeholder, ""), p


result, pp = manacher("babcbabcbaccba")
print(result, pp)

result, pp = manacher("abbaabbba")
print(result, pp)