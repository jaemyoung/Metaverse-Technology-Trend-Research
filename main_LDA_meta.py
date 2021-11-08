import numpy as np
import gensim
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim import corpora
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import re

now = '211020_' # 오늘날짜
file_path ='C:/Users/82109/GitHub/Metaverse_Technology_Trend_Research/data'

#preprocessing 완료된 document pickle 파일 열기
with open('data/preprocessing_data(2042)_title_abstract.pickle',"rb") as fr:
          tokenized_doc = pickle.load(fr)

# LDA 구현 

dictionary = corpora.Dictionary(tokenized_doc) # tokenized 데이터를 통해 dictionary로 변환
dictionary.filter_extremes(no_below=5, no_above=0.5) # 출현빈도 적거나 많으면 삭제
corpus = [dictionary.doc2bow(text) for text in tokenized_doc] # 코퍼스 구성
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 20, id2word=dictionary, iterations= 600, passes = 30, random_state = 1004 ) 

def make_topic_model(ldamodel,num_topic,num_word):
    
    topics = ldamodel.print_topics(num_topics= num_topic, num_words= num_word)
    df_topic = pd.DataFrame(topics)
    #단어 토큰화 작업 및 불필요 텍스트 제거
    tokenized_topic = [nltk.word_tokenize(doc.lower()) for doc in df_topic[1]]
    clean_topics= []
    for word in tokenized_topic:
        list_par = []
        for i in word:
            text = re.sub('[^a-zA-Z]',' ',i).strip() # 영어제외 다 제거.
            if(text != ''): # 영어,숫자 및 공백 제거.
                list_par.append(text)
        clean_topics.append(list_par)

    #DataFrame으로 틀만들어서 넣기    
    clean_topics= pd.DataFrame(clean_topics)
    df_topics =clean_topics.transpose()
    df_topics.columns = ['Topic'+str(i) for i in range(1,num_topic+1)]
                                                   
    # LDA modeling 결과 csv 파일 저장
    modeling_name = file_path + now+'Topic=' + str(num_topic)+ '_modeling.csv'
    df_topics.to_csv(modeling_name, index=True)
    print("make topic model complete!")
 

# Topic table 적용
def make_topic_table(ldamodel, corpus, num_topic):
    topictable = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topictable = topictable.append(pd.Series([int(topic_num), round(prop_topic,10), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
    topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
           
    #LDA table 결과 csv 저장
    table_name = file_path + now + 'Topic=' + str(num_topic)+ '_table.csv'
    topictable.to_csv(table_name, index=True)
    print("make topic table complete!")

# 문서별 토픽 유사도+ visualizer
def make_topic_simliarity(ladmodel,corpus,num_topic, num_cluster):
    simliarity_vetor=[]
    for i in range(len(corpus)):
        r=[]
        for w in ldamodel.get_document_topics(corpus[i], minimum_probability=0):
            r.append(w[1])
        simliarity_vetor.append(r)
    E= pd.DataFrame(simliarity_vetor)
    E.to_csv(file_path +now+ 'Topic='+str(num_topic)+'_simliarity.csv', header= ["topic"+str(i) for i in range(1, num_topic+1)])
    print("make topic simliarity complete!")
    
    kmeans = KMeans(n_clusters= num_cluster).fit(simliarity_vetor)
    clusters = kmeans.labels_
    TSNE_vetor = TSNE(n_components=2).fit_transform(simliarity_vetor)# component = 차원
    Q = pd.DataFrame(TSNE_vetor) # dataframe으로 변경하여 K-means cluster lavel 열 추가
    Q["clusters"] = clusters #lavel 추가
    fig, ax = plt.subplots(figsize=(24,15))
    sns.scatterplot(data = Q, x=0, y=1, hue= clusters, palette='deep')
    plt.show()
    print("visualizer complete!")
    
     
    
###########################################################################
# 적용(LDA 구현했을 때 파라미터 동일하게 해야됨)
make_topic_model(ldamodel, num_topic=20, num_word=20)
make_topic_table(ldamodel, corpus, num_topic =45)   
make_topic_simliarity(ldamodel,corpus,num_topic= 45, num_cluster = 45)
###########################################################################

# 빈도수 그래프
df = pd.read_csv("C:/Users/82109/GitHub/Metaverse_Technology_Trend_Research/data/210928_Topic=45_table.csv")
a= df["가장 비중이 높은 토픽"].value_counts()
ax = a.plot(kind='bar', title='Number of documents per topic', figsize=(12, 4), legend=None)
ax.set_xlabel('Topics', fontsize=12)          # x축 정보 표시
ax.set_ylabel('Number of documents', fontsize=12)     # y축 정보 표시

###########################################
# clustering 빈도수 그래프
num_topic = 40
cluster_n = 20
simliarity_vetor=[]
for i in range(len(corpus)):
    r=[]
    for w in ldamodel.get_document_topics(corpus[i], minimum_probability=0):
        r.append(w[1])
    simliarity_vetor.append(r)
E= pd.DataFrame(simliarity_vetor)
E.to_csv(file_path +now+ 'Topic='+str(num_topic)+'_simliarity.csv', header= ["topic"+str(i) for i in range(1, num_topic+1)])
print("make topic simliarity complete!")

simliarity_vetor.insert(cluster,0)
kmeans = KMeans(n_clusters= 40).fit(simliarity_vetor)
clusters = kmeans.labels_

# 빈도수 그래프
clusters = pd.DataFrame(clusters)
a= clusters[0].value_counts()

ax = a.plot(kind='bar', title='Number of cluster', figsize=(12, 4), legend=None)
ax.set_xlabel('Cluster', fontsize=12)          # x축 정보 표시
ax.set_ylabel('Number of documents', fontsize=12)     # y축 정보 표시