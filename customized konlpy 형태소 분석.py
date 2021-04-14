###url 데이터 불러오고 파싱하여 기사의 본문 가져오기
data= pd.read_excel('C:/Users/USER/Desktop/1~8newsurl/covid_url38_word.xlsx', index_col=0)
data = data.dropna(axis=0)
data= data[['date','title','text','category','url']]
#테스트용
#data = data[34700:35000]#본문 가져오기


###배제어 포함어를 추가한 okt를 사용해서 형태소 분석하기
twi = Twitter()
add_noun = ['비대면','확진자','거리두기','사랑제일교회','의료기관','의료진','집단감염','유연근무','유연근무제','유연근로제','한국판뉴딜','디지털뉴딜',
            '그린뉴딜','휴먼뉴딜','취약계층','긴급재난지원금','고용보험','진료거부','택배기사','선별진료소','고용보험','특수고용직','특고','재배포',
            '고용안정지원금','공유오피스','공용오피스','저소득층','실업급여','유통업계','물류센터','소상공인','유연근로','특수형태근로자','특수형태근로종사자']
eco_word = pd.read_excel('C:/Users/USER/Desktop/★주요키워드 및 배제어.xlsx')
eco_keyword = list(eco_word['경제키워드'])
for i in eco_keyword:
    add_noun.append(i)

#새로운 단어 등록 하기
for i in add_noun:
    twi.add_dictionary(i,'Noun')

#통합어 만들기
replace = {'유연근로제':'유연근무제','유연근무':'유연근무제','유연근로':'유연근무제','특고':'특수고용직','특수형태근로자':'특수고용직','특수형태근로종사자':'특수고용직'}
postprocessor = Postprocessor(base_tagger=twi, replace=replace, passtags={'Noun'})

#단어 등록 확인하기
postprocessor.pos('확진자와 비대면으로 거리두기를 사랑제일교회에서부터 전광훈이 사전조사 의료기관과 의료진이 집단감염이 걸려 집단감염, 재택근무하고 '
          '유연근무와 유연근무제를 하고 지역경제와 한국판뉴딜과 그린뉴딜과 디지털뉴딜을 휴면뉴딜하고 취약계층의 긴급재난지원금을 특수고용직에게'
          '고용보험을 진료거부하며 택배기사는 온라인수업으로 선별진료소로 간다 소상공인 갭투자를 계액갱신공매공매도공시가격과징금과학기술괴리율국가경쟁력국내총생산'
          '국민건강보험국민연금국제원유가격규제지역그린뉴딜그린벨트금리금리인하금융기관금융세제금융위기금융지원배달서비스배당금배터리백신보유세보이스피싱부동산부동산시장'
          '분양가분양가상한제불법사금융블록체인비상경제비상경제회의빅데이터사모펀드사이드카사회적거리두기산유국상장주식상장지수증권상장지수펀드생필품서킷브레이커선물세트선물환세금'
          '세입자셧다운소비심리소상공인손소독제수소차수출규제스마트스토어스마트팜스타트업시장심리지수신용대출신용등급신용카드신재생에너지실물경제실업자안전자산액화천연가스양도세양도소득양도소득세'
          '양도차익양적완화에너지저장시스템오픈뱅킹온라인쇼핑몰외환시장외환위기용적률우선주운전자보험원격수업원격의료원달러환율위안화 AI에서 특수형태근로종사자와 특수형태근로자와 특고')

#형태소 분석 하기
words = [[j[i][0] for i in range(len(j))] for j in [postprocessor.pos(i) for i in tqdm(list(data['text']))]]

#배제어 등록하기
Prohibit_words = ['기자','연합뉴스','뉴시스','시사저널','신문','뉴스','사진','헤럴드경제','노컷뉴스','파이낸셜뉴스','특파원',
                  '라며','대해','지난','위해','오전','오후','무단','배포','이데일리','머니투데이','앵커','지금','때문','이번',
                  '통해','정도','경우','관련','이미지','출처','일보','바로가기','까지','여개','도록','이나','재배포','처럼','면서',
                  '거나','이제','지난달','어요']
for i in list(eco_word['배제어']):
    Prohibit_words.append(i)
#배제어 제거, 한 글자 제거하기
j = 0
for i in tqdm(words):
    for k in Prohibit_words:
        while k in i:
            i.remove(k)
    words[j] = i
    j += 1 #불용어 제외

for k in range(len(words)):
    words[k] = [i for i in words[k] if len(i) > 1]  # 한글자 제외

data['words'] = words

data['length_word'] = [len(i) for i in tqdm(data['words'])]

data50 = data[data['length_word'] >= 50]  #단어 갯수 50개 이하 제거
data50.to_excel('C:/Users/USER/Desktop/1~8newsurl/data0921.xlsx',header=True) #저장하기

#3글자 부터 밎 주요 2단어 포함 토픽모델링용
data = data50.reset_index(drop=True)
words = list(data['words'])
for k in range(len(words)):
    words[k] = [i for i in words[k] if len(i) > 2 or i in ['배달','택배','고용','취업','실업','채용','구직','청년','유통','해고']]  # 두글자 이하 제외
data['words'] = words
data['words'][:5]
del words; del k;