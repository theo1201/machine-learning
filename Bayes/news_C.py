import pandas as pd
import jieba
import numpy
# coding: utf-8

# ###jieba特性介绍
# 支持三种分词模式：
# 精确模式，试图将句子最精确地切开，适合文本分析；
# 全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
# 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
# 支持繁体分词。
# 支持自定义词典。
# MIT 授权协议。
# ###分词速度
# 1.5 MB / Second in Full Mode
# 400 KB / Second in Default Mode
# jieba.cut 方法接受三个输入参数: 需要分词的字符串；cut_all 参数用来控制是否采用全模式；HMM 参数用来控制是否使用 HMM 模型。
# jieba.cut_for_search 方法接受两个参数：需要分词的字符串；是否使用 HMM 模型。该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细。
# 待分词的字符串可以是 unicode 或 UTF-8 字符串、GBK 字符串。注意：不建议直接输入 GBK 字符串，可能无法预料地错误解码成 UTF-8。
# jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语(unicode)，或者用
# jieba.lcut 以及 jieba.lcut_for_search 直接返回 list。
# jieba.Tokenizer(dictionary=DEFAULT_DICT) 新建自定义分词器，可用于同时使用不同词典。jieba.dt 为默认分词器，所有全局分词相关函数都是该分词器的映射。


df_news = pd.read_table('./data/val.txt',names=['category','theme','URL','content'],encoding='utf-8')
df_news = df_news.dropna()
print(df_news.head())
print(df_news.shape)
# 使用结巴分词器
content = df_news.content.values.tolist()
print (content[1000])
content_S = []
for line in content:
	current_segment = jieba.lcut(line)
	if len(current_segment)>1 and current_segment !='\r\n':
		content_S.append(current_segment)

# print(content_S[1000])
df_content=pd.DataFrame({'content_S':content_S})
# df_content.head()
stopwords=pd.read_csv("stopwords.txt",index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8')
# stopwords.head(20)

def drop_stopwords(contents, stopwords):
	contents_clean = []
	all_words = []
	for line in contents:
		line_clean = []
		for word in line:
			if word in stopwords:
				continue
			line_clean.append(word)
			all_words.append(str(word))
		contents_clean.append(line_clean)
	return contents_clean, all_words


# print (contents_clean)
contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords)

df_content=pd.DataFrame({'contents_clean':contents_clean})
# df_content.head()

df_all_words=pd.DataFrame({'all_words':all_words})
# df_all_words.head()

words_count=df_all_words.groupby(by=['all_words'])['all_words'].agg({"count":numpy.size})
words_count=words_count.reset_index().sort_values(by=["count"],ascending=False)
# words_count.head()

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
#
# wordcloud=WordCloud(font_path="./data/simhei.ttf",background_color="white",max_font_size=80)
# word_frequence = {x[0]:x[1] for x in words_count.head(100).values}
# wordcloud=wordcloud.fit_words(word_frequence)
# plt.imshow(wordcloud)

# 提取关键词TF-IDF
import jieba.analyse
index = 2400
print (df_news['content'][index])
content_S_str = "".join(content_S[index])
print ("  ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))

# LDA 主题模型
from gensim import corpora, models, similarities
import gensim
#做映射，相当于词袋
dictionary = corpora.Dictionary(contents_clean)
corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean]

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20) #类似Kmeans自己指定K值
#一号分类结果
print (lda.print_topic(1, topn=5))
for topic in lda.print_topics(num_topics=20, num_words=5):
    print (topic[1])
df_train=pd.DataFrame({'contents_clean':contents_clean,'label':df_news['category']})
df_train.tail()
df_train.label.unique()

label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育":5, "教育": 6,"文化": 7,"军事": 8,"娱乐": 9,"时尚": 0}
df_train['label'] = df_train['label'].map(label_mapping)
df_train.head()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, random_state=1)

words = []
for line_index in range(len(x_train)):
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        words.append(' '.join(x_train[line_index]))
    except:
        # print (line_index,word_index)
		print(line_index)

from sklearn.feature_extraction.text import CountVectorizer
texts=["dog cat fish","dog cat cat","fish bird", 'bird']
cv = CountVectorizer()
cv_fit=cv.fit_transform(texts)

print(cv.get_feature_names())
print(cv_fit.toarray())


print(cv_fit.toarray().sum(axis=0))

from sklearn.feature_extraction.text import CountVectorizer
texts=["dog cat fish","dog cat cat","fish bird", 'bird']
cv = CountVectorizer(ngram_range=(1,4))
cv_fit=cv.fit_transform(texts)

print(cv.get_feature_names())
print(cv_fit.toarray())


print(cv_fit.toarray().sum(axis=0))

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(analyzer='word', max_features=4000,  lowercase = False)
vec.fit(words)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(words), y_train)

test_words = []
for line_index in range(len(x_test)):
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        test_words.append(' '.join(x_test[line_index]))
    except:
         # print (line_index,word_index)
		print(line_index)
test_words[0]

classifier.score(vec.transform(test_words), y_test)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='word', max_features=4000,  lowercase = False)
vectorizer.fit(words)

# from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)

classifier.score(vectorizer.transform(test_words), y_test)


