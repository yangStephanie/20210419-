#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import jieba
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

pd.set_option('display.max_colwidth', None)#setting the maximize string show

fb = pd.read_csv('nysu_10902_2019立委_research.csv')
politics = pd.read_csv('9th_legislator_promise.csv')


# In[4]:


kuan = fb[fb.page_name == "管碧玲 (kuanbiling)"]
# 把時間格式轉換
kuan['new_date'] = pd.to_datetime(kuan['created_time_taipei']).dt.date
kuan['post_hour'] = pd.to_datetime(kuan['created_time_taipei']).dt.hour #找出發文 '時間'(hour)
kuan['month_year'] = pd.to_datetime(kuan['new_date']).dt.to_period('M')

kuan.head(1)


# In[6]:


kuan[kuan.created_time_taipei == max(kuan.created_time_taipei)]


# In[7]:


kuan[kuan.created_time_taipei == min(kuan.created_time_taipei)]


# In[9]:


kuan = kuan.sort_values(by=['like_count'])
kuan


# In[10]:


kuan.reset_index(inplace=True)
kuan


# 結巴斷詞 jieba
# 
# len(liu): 看劉櫂豪的貼文總共有幾筆
# list(liu['message']): 把要斷詞的內容轉成 list 格式
# 創造一個新的dataframe叫做liu_docs，型態是pandas.core.frame.DataFrame
# 原本liu_docs裡面的'jieba_results'欄位型態是pandas.core.series.Series，需要轉成'str'(字串)型態才能做後續處理。

# In[11]:


doc_names = range(len(kuan)) #管碧玲有幾篇文章
doc_names


# In[48]:


import numpy as np
kuan = kuan.replace(np.nan, '', regex=True) #將nan取代成空白字串

text_list = list(kuan['message'])
kuan_docs = pd.DataFrame(columns=['jieba_results'])
kuan_docs['jieba_results'] = kuan_docs['jieba_results'].astype('str')


# In[49]:


words = jj.lcut(text_list[1], cut_all = False)
print(words)


# In[50]:


words = jj.lcut(text_list[527], cut_all = False)
print(words)


# 把要存斷詞結果的dataframe準備好之後，就可以開始斷詞了。</br>
# 
# 第一個迴圈</br> 剛才我們知道，劉櫂豪的篇數有999篇： 我們可以用一個for迴圈來跑每一篇文章，i是對應文章的意思，會從1(第一篇)跑到999(第999篇)；</br> 而text_list就是我們剛剛轉成list型態的劉櫂豪貼文內容；text_list[1]，就是取出list第一個元素(第一篇貼文)的意思。</br>
# 
# 第二個迴圈</br> words存的是文章斷詞後的集合物件(object)，我們需要跑第二個for迴圈把集合裡面的字取出來，</br> word代表斷詞集合中的一個字，先把字轉換成str(字串)型態之後，字和字之間加上空格存在一起。
# 
# append加回斷詞欄位</br> 之後把text內容存到'jieba_results'這個欄位裡面，再和liu_docs合併。</br> 最後我們liu_docs，只會一個欄位jieba_results，存斷詞之後的結果。

# In[51]:


import jieba
from tqdm import tqdm
import re
jieba.load_userdict("dict.txt")

punctuation = "、，：:""()\n!！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
re_punctuation = "[{}] ".format(punctuation)

for i in doc_names: #從第一篇到最後一篇
    words = jieba.cut(text_list[i], cut_all = False)
    text = ''
    for word in words:
        text = text + ' ' + str(word) 
    #print(words)
    #print(i) #第幾篇文章
    #print(text) #第幾篇文章的斷詞結果
    text = re.sub(re_punctuation, "", text)
    text = re.sub(r'[0-9]','',text)
    text = re.sub(r'[a-zA-Z]','',text)
    s = pd.Series({'jieba_results': text})
    kuan_docs = kuan_docs.append(s, ignore_index=True)#每次的斷詞結果都加回kuan_docs


# In[52]:


kuan_docs


# In[53]:


kuan['jieba_results'] = kuan_docs
kuan_post = kuan[['new_date','message','jieba_results','like_count']]
kuan_post


# In[54]:


import matplotlib.pyplot as plt


# In[55]:


commutes = kuan_post['like_count']
commutes.plot.hist(grid=True, bins=100, rwidth=0.9,
color='#607c8e')
plt.grid(axis='y', alpha=0.75)


# In[56]:


temp = kuan_post[(kuan_post['like_count'] <= 1000) & (kuan_post['like_count'] > 500)]
temp


# In[57]:


corpus = temp['jieba_results'].values.tolist()


# In[58]:


# TF-IDF
# coding:utf-8  

#算字頻
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer = CountVectorizer()  
X = vectorizer.fit_transform(corpus)  
word = vectorizer.get_feature_names()  

#計算TFIDF
from sklearn.feature_extraction.text import TfidfTransformer  
transformer = TfidfTransformer()  
tfidf = transformer.fit_transform(X)


# In[59]:


from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=8, random_state=None)
LDA.fit(tfidf)

#觀看結果
for i,topic in enumerate(LDA.components_):
    print(f"TOP 10 WORDS PER TOPIC #{i}")
    print([vectorizer.get_feature_names()[index] for index in topic.argsort()[-10:]])


# In[60]:


LDA.fit(X)

#觀看結果
for i,topic in enumerate(LDA.components_):
    print(f"TOP 10 WORDS PER TOPIC #{i}")
    print([vectorizer.get_feature_names()[index] for index in topic.argsort()[-10:]])


# In[61]:


politics_kuan = politics[politics.姓名 == "管 碧 玲"].政見.to_string()
politics_kuan = re.sub(re_punctuation, "", politics_kuan)
politics_kuan = re.sub(r'[0-9]','',politics_kuan)
politics_kuan = re.sub(r'[a-zA-Z]','',politics_kuan)
jieba.load_userdict("dict.txt")
words = jieba.lcut(politics_kuan, cut_all = False)
print(words)


# In[99]:


vectorizer = CountVectorizer()  
X = vectorizer.fit_transform(words)  
word = vectorizer.get_feature_names() 
transformer = TfidfTransformer()  
tfidf = transformer.fit_transform(X)

LDA.fit(tfidf)

#觀看結果
for i,topic in enumerate(LDA.components_):
    print(f"TOP 10 WORDS PER TOPIC #{i}")
    print([vectorizer.get_feature_names()[index] for index in topic.argsort()[-10:]])


# In[121]:


kuan = fb[fb.page_name == "管碧玲 (kuanbiling)"]
# 把時間格式轉換
kuan['new_date'] = pd.to_datetime(kuan['created_time_taipei']).dt.date
kuan['post_hour'] = pd.to_datetime(kuan['created_time_taipei']).dt.hour #找出發文 '時間'(hour)
kuan['month_year'] = pd.to_datetime(kuan['new_date']).dt.to_period('D')



juf=kuan[(kuan['month_year']>'2019-01-01')&(kuan['month_year']<'2019-07-31')]
juf



# In[122]:


doc_names = range(len(juf)) #管碧玲有幾篇文章
doc_names


# In[123]:


import numpy as np
juf = juf.replace(np.nan, '', regex=True) #將nan取代成空白字串

text_list = list(juf['message'])
juf_docs = pd.DataFrame(columns=['jieba_results'])
juf_docs['jieba_results'] = juf_docs['jieba_results'].astype('str')


# In[124]:


words = jj.lcut(text_list[1], cut_all = False)
print(words)


# In[127]:


words = jj.lcut(text_list[234], cut_all = False)
print(words)


# In[ ]:




