import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

# 글씨체
font_name = font_manager.FontProperties(fname='C:/Windows/Fonts/H2HDRM.TTF').get_name()
rc('font', family=font_name)

# 데이터 불러오기 각각 19년 20년 기사
data2019 = pd.read_excel('2019gisa.xlsx')
data2020 = pd.read_excel('2020gisa.xlsx')

data2019['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
                     list(data2019['words'])]  # words가 텍스트 형식으로 되어 있을 경우
data2020['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
                     list(data2020['words'])]  # words가 텍스트 형식으로 되어 있을 경우

# 분석대상 키워드 (분석대상 키워드는 통계청 뉴스기반 검색사이트를 사용하여 임의로 정했음)
keyword_list = ['유통', '청년', '근로자', '노동자', '배달', '재택근무',
                '화상회의', '택배', '출퇴근', '유연근무제']

# 키워드별 기사량 증감율 확인하기
# 년도별 키워드별 기사량 확인하기
count2019 = []
for k in keyword_list:
    j = 0
    for i in data2019['words']:
        if k in i:
            j += 1
    count2019.append(j)

count2020 = []
for k in keyword_list:
    j = 0
    for i in data2020['words']:
        if k in i:
            j += 1
    count2020.append(j)

keyword_data = pd.DataFrame({'keyword': keyword_list, 'count2019': count2019, 'count2020': count2020})
keyword_data['rate'] = [(i - j) / j for i, j in zip(count2020, count2019)]  # 증가율 계산하기

# 그래프를 그려 증감율 확인하기
plt.plot()
g = sns.barplot(x=keyword_data['keyword'], y=keyword_data['rate'])
for i, j in enumerate(keyword_data['rate']):
    g.text(x=i - 0.3, y=j, s="{:.1f}%".format(j))

# 키워드에 따른 뉴스분류를 사전방식과 word2vec 방법을 이용
# 사전 방식으로 뉴스 분류 (포함어 방식)
for i in keyword_list:
    data2019[i] = 0
    data2020[i] = 0

for i in keyword_list:
    for j in range(len(data2019)):
        if i in data2019['words'][j]:
            data2019[i][j] = 1
    for j in range(len(data2020)):
        if i in data2020['words'][j]:
            data2020[i][j] = 1

# 잘 되었는지 확인
data2019['청년'].value_counts()
data2020['청년'].value_counts()

# 기사 단위로 word2vec 분석하기
embedding_model = Word2Vec(pd.concat([data2020['words'], data2019['words']]), vector_size=100, window=5, min_count=30,
                           workers=4, epochs=50)

embedding_model.wv.most_similar('유통', topn=10)


# 기사들이 해당 키워드와 얼마나 관련이 있는지 점수를 매기는 함수 word2vec 유사도 점수를 활용
def scoring(word, keyword):
    avg_dist = []
    dist = []
    dist_dist = []
    for i in tqdm(range(len(word))):
        for k in word[i]:
            try:
                # 비교하여 similarity 구하기
                dist_dist.append(embedding_model.wv.similarity(keyword, k))
            except:
                dist_dist.append(0)
            dist.append(np.array(dist_dist))
            dist_dist = []
        avg_dist.append(np.mean(dist))
        dist = []
    return avg_dist


for i in keyword_list:
    globals()['data2019' + i] = data2019[data2019[i] == 1].reset_index(drop=True)
    globals()['data2019' + i]['score'] = scoring(globals()['data2019' + i]['words'], i)
    globals()['data2020' + i] = data2020[data2020[i] == 1].reset_index(drop=True)
    globals()['data2020' + i]['score'] = scoring(globals()['data2020' + i]['words'], i)


