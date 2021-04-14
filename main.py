import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
from matplotlib import font_manager, rc
# 글씨체
font_name = font_manager.FontProperties(fname='C:/Windows/Fonts/H2HDRM.TTF').get_name()
rc('font', family=font_name)


# 데이터 불러오기 각각 19년 20년 기사
data2019 = pd.read_excel('2019gisa.xlsx')
data2020 = pd.read_excel('2020gisa.xlsx')

data2019['words'] = [i.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(',') for i in
          list(data2019['words'])]  # words가 텍스트 형식으로 되어 있을 경우

# 분석 속도를 올리기 위해 일자별 샘플링
def sampling_func(data777, sample_pct):
    np.random.seed(123)
    N = len(data777)
    sample_n = int(len(data777)*sample_pct) # integer
    sample = data777.take(np.random.permutation(N)[:sample_n])
    return sample

data2019 = data2019.groupby('date', group_keys=False).apply(sampling_func,sample_pct=0.2)  #일자별 샘플링
data2020 = data2020.groupby('date', group_keys=False).apply(sampling_func,sample_pct=0.2)

data2019.groupby(data2019['date']).size()  # size 확인


###형태소 분석한것을 이용하여 기사 단위로 word2vec 분석하기
embedding_model = Word2Vec(data2019['words'], vector_size=50, window=5, min_count=10, workers=4, epochs=50)

embedding_model.wv.most_similar('고용',topn=10)

def scoring(word):
    avg_dist = []
    dist = []
    dist_dist = []
    for i in tqdm(range(len(word))):
        for k in word[i]:
            try:
                 # 비교하여 similarity 구하기
                dist_dist.append(embedding_model.similarity('고용', k))
            except:
                dist_dist.append(0)
            dist.append(np.array(dist_dist))
            dist_dist = []
        avg_dist.append(np.mean(dist))
        dist = []
    return avg_dist