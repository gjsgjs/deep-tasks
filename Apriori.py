def load_data():
    # 示例数据集
    return [
        ['Milk', 'Bread', 'Butter'],
        ['Beer', 'Bread'],
        ['Milk', 'Bread', 'Butter', 'Beer'],
        ['Milk', 'Butter'],
        ['Bread', 'Butter'],
    ]


# 返回单项集
def create_candidates(dataset):
    """
       Aprior 创建单项候选集C1，即所有可能的一项集
       @dataset: 数据
       Return 单项集 C1
    """
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # 候选项集需要使用 frozenset 而不是 set，因为 frozenset 是不可变的并且是可哈希的
    return list(map(frozenset, C1))


def scan_dataset(dataset, candidates, min_support):
    """
          @dataset: 数据
          @candidates: 候选数据集Ck
          @support_count: 项集的频度, 最小支持度计数阈值
          Return Lk 满足支持度计数项集
    """
    ss_cnt = {}
    for tid in dataset:
        for can in candidates:
            # 记录候选集出现的个数
            if can.issubset(tid):
                if not can in ss_cnt:
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    num_items = float(len(dataset))
    ret_list = []
    support_data = {}
    for key in ss_cnt:
        support = ss_cnt[key] / num_items
        # 频繁项集
        if support >= min_support:
            ret_list.insert(0, key)
        support_data[key] = support
    return ret_list, support_data


def apriori_gen(freq_sets, k):
    ret_list = []
    len_freq_sets = len(freq_sets)
    # 连接步 (k-1)><(k) 确保合并的集合前k-2项相同
    for i in range(len_freq_sets):
        for j in range(i + 1, len_freq_sets):
            L1 = list(freq_sets[i])[:k - 2]
            L2 = list(freq_sets[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                # 取并集
                ret_list.append(freq_sets[i] | freq_sets[j])
    return ret_list


def apriori(dataset, min_support=0.5):
    D = list(map(set, dataset))
    C1 = create_candidates(dataset)
    L1, support_data = scan_dataset(D, C1, min_support)
    L = [L1]
    # 项集元素数
    k = 2
    while len(L[k - 2]) > 0:
        Ck = apriori_gen(L[k - 2], k)
        if Ck == []:
            break
        Lk, supK = scan_dataset(D, Ck, min_support)
        support_data.update(supK)
        L.append(Lk)
        k += 1
    return L, support_data


if __name__ == "__main__":
    dataset = load_data()
    L, support_data = apriori(dataset, min_support=0.4)
    for i, freq_sets in enumerate(L):
        print(f"频繁 {i + 1}-项集：")
        for freq_set in freq_sets:
            print(freq_set, "支持度：", support_data[freq_set])
