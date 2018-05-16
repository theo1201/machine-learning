import re, collections
# argmaxc P(c|w) -> argmaxc P(w|c) P(c) / P(w)
# P(c), 文章中出现一个正确拼写词 c 的概率, 也就是说, 在英语文章中, c 出现的概率有多大
# P(w|c), 在用户想键入 c 的情况下敲成 w 的概率. 因为这个是代表用户会以多大的概率把 c 敲错成 w
# argmaxc, 用来枚举所有可能的 c 并且选取概率最大的
# 将所有的单词变成小写，并去掉特殊字符，返回
def words(text): return re.findall('[a-z]+', text.lower())

def train(features):
	# 因为最后需要乘，所以预先设置如果没有出现的单词，设置他出现1次
	model = collections.defaultdict(lambda: 1)
	# 遍历数据集，返回频率分布数组
	for f in features:
		model[f] += 1
	return model


NWORDS = train(words(open('big.txt').read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'
# 编辑距离插入，删除，交换，替换，最少经过的变换次数
# 返回所有与单词w编辑距离为1的集合
def edits1(word):
	n = len(word)
	return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion，删
			   [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition，交换
			   [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration，替换
			   [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion 插入

# 因为编辑距离1返回的数据集比较大，所以在距离1的基础上进行距离2的查找

def known_edits2(word):
	return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)
# 优化
#返回所有与单词 w 编辑距离为 2 的集合
#在这些编辑距离小于2的词中间, 只把那些正确的词作为候选词
def edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))

# 查找对应的词
def known(words): return set(w for w in words if w in NWORDS)

#如果known(set)非空, candidate 就会选取这个集合, 而不继续计算后面的
def correct(word):
	# 指定一个优先级，按照编辑距离进行查找返回
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=lambda w: NWORDS[w])

print(correct('knn'))