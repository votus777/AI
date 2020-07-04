import pandas as pd
import numpy as np
from pyarrow import csv as pycsv

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


font_location = '\\Users\\bitcamp\\font\\NanumBarunGothic.ttf'
font_name = fm.FontProperties(fname=font_location, size=9).get_name()

mpl.rc('font',family = font_name)
mpl.font_manager._rebuild()

import seaborn as sns

import warnings
warnings.filterwarnings(action = 'ignore')

# 데이터 
data = pycsv.read_csv('\\Users\\bitcamp\\Documents\\GitHub\\AI_study\\data\\dacon_data\\comp_jeju\\train.csv').to_pandas()
sub = pycsv.read_csv('\\Users\\bitcamp\\Documents\\GitHub\\AI_study\\data\\dacon_data\\comp_jeju\\submission.csv').to_pandas()

sample = data.iloc[ : 60000]
sample = sample.drop(['CARD_CCG_NM','HOM_CCG_NM'], axis=1)
sample.to_csv('\\Users\\bitcamp\\Documents\\GitHub\\AI_study\\\data\dacon_data\\comp_jeju\\sample.csv', encoding='utf-8-sig')


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


reduce_mem_usage(data, verbose=True)
# Mem. usage decreased to 1648.76 Mb (27.1% reduction)




'''

data.info()



2019년 1월 ~ 2020년 3월 데이터로 

2020년 04월 ~ 07월 이용 금액 예측하기 


<class 'pandas.core.frame.DataFrame'>      
RangeIndex: 24697792 entries, 0 to 24697791  data_shape = (24697792, 12)
Data columns (total 12 columns):

 #   Column        Dtype   isna()
---  ------        -----   ----     
 0   REG_YYMM      int64    0
 1   CARD_SIDO_NM  object   0        카드이용지역_시도 (가맹점 주소 기준)
 2   CARD_CCG_NM   object   87213    카드이용지역_시군구 (가맹점 주소 기준)
 3   STD_CLSS_NM   object   0        업종명
 4   HOM_SIDO_NM   object   0        거주지역_시도 (고객 집 주소 기준)    
 5   HOM_CCG_NM    object   147787   거주지역_시군구 (고객 집 주소 기준)
 6   AGE           object   0        나이
 7   SEX_CTGO_CD   int64    0        성별  ( 남성 : 1, 여성 : 2)
 8   FLC           int64    0        가구생애주기  (1: 1인가구, 2: 영유아자녀가구, 3: 중고생자녀가구, 4: 성인자녀가구, 5: 노년가구)
 9   CSTMR_CNT     int64    0        이용 고객수 
 10  AMT           int64    0        이용 금액    -> Target
 11  CNT           int64    0        이용 건수 
 

 #중복된 행의 데이터만 표시하기
check_data = data[data.columns[:-3]]
display(check[check_data.duplicated()])
중복 데이터 없음 

# categorical data 확인 
for i in range(len(data)) : 
    
    unique_colums = data.columns[i]
    unique = data.iloc[ : , i].unique()
    
    print(unique_colums, unique)



CARD_SIDO_NM  -  카드이용지역_시도 (가맹점 주소 기준)
['강원' '경기' '경남' '경북' '광주' '대구' '대전' '부산' '서울' '세종' '울산' '인천' '전남' '전북' '제주' '충남' '충북'] - 17개 
 
 
CARD_CCG_NM   - 카드이용지역_시군구 (가맹점 주소 기준)
['강릉시' '고성군' '동해시' '삼척시' '속초시' '양구군' '양양군' '영월군' '원주시' '인제군' '정선군' '철원군'
 '춘천시' '태백시' '평창군' '홍천군' '화천군' '횡성군' '가평군' '고양시 덕양구' '고양시 일산동구' '고양시 일산서구'
 '과천시' '광명시' '광주시' '구리시' '군포시' '김포시' '남양주시' '동두천시' '부천시' '성남시 분당구'
 '성남시 수정구' '성남시 중원구' '수원시 권선구' '수원시 영통구' '수원시 장안구' '수원시 팔달구' '시흥시'
 '안산시 단원구' '안산시 상록구' '안성시' '안양시 동안구' '안양시 만안구' '양주시' '양평군' '여주시' '연천군'
 '오산시' '용인시 기흥구' '용인시 수지구' '용인시 처인구' '의왕시' '의정부시' '이천시' '파주시' '평택시' '포천시'
 '하남시' '화성시' '거제시' '거창군' '김해시' '남해군' '밀양시' '사천시' '산청군' '양산시' '의령군' '진주시'
 '창녕군' '창원시 마산합포구' '창원시 마산회원구' '창원시 성산구' '창원시 의창구' '창원시 진해구' '통영시' '하동군'
 '함안군' '함양군' '합천군' '경산시' '경주시' '고령군' '구미시' '군위군' '김천시' '문경시' '봉화군' '상주시'
 '성주군' '안동시' '영덕군' '영양군' '영주시' '영천시' '예천군' '울릉군' '울진군' '의성군' '청도군' '청송군'
 '칠곡군' '포항시 남구' '포항시 북구' '광산구' '남구' '동구' '북구' '서구' '달서구' '달성군' '수성구' '중구'
 '대덕구' '유성구' '강서구' '금정구' '기장군' '동래구' '부산진구' '사상구' '사하구' '수영구' '연제구' '영도구'
 '해운대구' '강남구' '강동구' '강북구' '관악구' '광진구' '구로구' '금천구' '노원구' '도봉구' '동대문구' '동작구'
 '마포구' '서대문구' '서초구' '성동구' '성북구' '송파구' '양천구' '영등포구' '용산구' '은평구' '종로구' '중랑구'
 nan '울주군' '강화군' '계양구' '남동구' '부평구' '연수구' '옹진군' '강진군' '고흥군' '곡성군' '광양시'
 '구례군' '나주시' '담양군' '목포시' '무안군' '보성군' '순천시' '신안군' '여수시' '영광군' '영암군' '완도군'
 '장성군' '장흥군' '진도군' '함평군' '해남군' '화순군' '고창군' '군산시' '김제시' '남원시' '무주군' '부안군'
 '순창군' '완주군' '익산시' '임실군' '장수군' '전주시 덕진구' '전주시 완산구' '정읍시' '진안군' '서귀포시'
 '제주시' '계룡시' '공주시' '금산군' '논산시' '당진시' '보령시' '부여군' '서산시' '서천군' '아산시' '예산군'
 '천안시 동남구' '천안시 서북구' '청양군' '태안군' '홍성군' '괴산군' '단양군' '보은군' '영동군' '옥천군' '음성군'
 '제천시' '증평군' '진천군' '청주시 상당구' '청주시 서원구' '청주시 청원구' '청주시 흥덕구' '충주시']
 
 
STD_CLSS_NM  - 업종명
['건강보조식품 소매업' '골프장 운영업' '과실 및 채소 소매업' '관광 민예품 및 선물용품 소매업'
 '그외 기타 스포츠시설 운영업' '그외 기타 종합 소매업' '기타 대형 종합 소매업' '기타 외국식 음식점업' '기타 주점업'
 '기타음식료품위주종합소매업' '마사지업' '비알콜 음료점업' '빵 및 과자류 소매업' '서양식 음식점업' '수산물 소매업'
 '슈퍼마켓' '스포츠 및 레크레이션 용품 임대업' '여관업' '욕탕업' '육류 소매업' '일반유흥 주점업' '일식 음식점업'
 '전시 및 행사 대행업' '중식 음식점업' '차량용 가스 충전업' '차량용 주유소 운영업' '체인화 편의점'
 '피자 햄버거 샌드위치 및 유사 음식점업' '한식 음식점업' '호텔업' '화장품 및 방향제 소매업' '휴양콘도 운영업' '여행사업'
 '자동차 임대업' '면세점' '버스 운송업' '택시 운송업' '기타 수상오락 서비스업' '내항 여객 운송업'
 '그외 기타 분류안된 오락관련 서비스업' '정기 항공 운송업']
 
 
HOM_SIDO_NM - 거주지역_시도 (고객 집 주소 기준)  
['강원' '경기' '서울' '경남' '경북' '대구' '대전' '세종' '인천' '충남' '충북' '광주' '부산' '울산' '전남' '전북' '제주']
 
 
HOM_CCG_NM  - 거주지역_시군구 (고객 집 주소 기준)
['강릉시' '속초시' '동해시' '춘천시' '평창군' '성남시 분당구' '안산시 단원구' '용인시 기흥구' '용인시 수지구'
 '강남구' '영월군' '원주시' '정선군' '홍천군' '횡성군' '가평군' '고양시 덕양구' '고양시 일산동구' '고양시 일산서구'
 '광명시' '광주시' '구리시' '군포시' '김포시' '남양주시' '부천시' '성남시 수정구' '성남시 중원구' '수원시 권선구'
 '수원시 영통구' '수원시 장안구' '수원시 팔달구' '시흥시' '안산시 상록구' '안성시' '안양시 동안구' '안양시 만안구'
 '양주시' '양평군' '여주시' '오산시' '용인시 처인구' '의왕시' '의정부시' '이천시' '파주시' '평택시' '포천시'
 '하남시' '화성시' '창원시 성산구' '상주시' '포항시 남구' '달서구' '대덕구' '서구' '유성구' '강동구' '강북구'
 '강서구' '관악구' '광진구' '구로구' '금천구' '노원구' '도봉구' '동대문구' '동작구' '마포구' '서대문구' '서초구'
 '성동구' '성북구' '송파구' '양천구' '영등포구' '용산구' '은평구' '종로구' '중구' '중랑구' nan '계양구'
 '남구' '남동구' '부평구' '연수구' '아산시' '천안시 동남구' '천안시 서북구' '괴산군' '제천시' '진천군'
 '청주시 상당구' '청주시 청원구' '청주시 흥덕구' '충주시' '포항시 북구' '북구' '고성군' '삼척시' '양양군' '인제군'
 '태백시' '김해시' '양산시' '창원시 마산합포구' '창원시 진해구' '경산시' '경주시' '구미시' '안동시' '울진군'
 '광산구' '달성군' '동구' '수성구' '금정구' '기장군' '동래구' '부산진구' '사하구' '해운대구' '목포시' '군산시'
 '익산시' '전주시 덕진구' '전주시 완산구' '제주시' '논산시' '당진시' '서산시' '음성군' '청주시 서원구' '과천시'
 '동두천시' '창원시 의창구' '영주시' '순천시' '연천군' '진주시' '사상구' '연제구' '계룡시' '공주시' '철원군'
 '강화군' '여수시' '울주군' '거제시' '봉화군' '영덕군' '단양군' '양구군' '화천군' '창원시 마산회원구' '김천시'
 '예천군' '칠곡군' '수영구' '영도구' '광양시' '서귀포시' '보령시' '홍성군' '사천시' '통영시' '영천시' '김제시'
 '서천군' '옥천군' '증평군' '울릉군' '화순군' '예산군' '완주군' '태안군' '나주시' '창녕군' '정읍시' '밀양시'
 '함안군' '담양군' '부안군' '부여군' '의성군' '문경시' '청송군' '영암군' '남원시' '금산군' '보은군' '산청군'
 '거창군' '함양군' '고령군' '청도군' '영동군' '무안군' '해남군' '하동군' '구례군' '완도군' '합천군' '성주군'
 '장성군' '고창군' '무주군' '남해군' '군위군' '신안군' '영광군' '임실군' '청양군' '보성군' '옹진군' '고흥군'
 '강진군' '진도군' '곡성군' '함평군' '영양군' '장흥군' '진안군' '순창군' '장수군' '의령군']
 
 
AGE ['20s' '30s' '40s' '50s' '60s' '70s' '10s']



'''


# # 업종별 등장 빈도수
# fig = plt.figure(figsize=(20, 15))
# fig.patch.set_facecolor('xkcd:mint green')
# sns.barplot(y=data['STD_CLSS_NM'].value_counts().index,x=data['STD_CLSS_NM'].value_counts())
# plt.tight_layout()
# plt.show()

# # 1등 한식 음식점업
# # 2등 편의점 


# city_count= data.groupby(['CARD_SIDO_NM','CARD_CCG_NM'])['STD_CLSS_NM'].value_counts().reset_index(name='count')
# city_sum = city_count.groupby(['CARD_SIDO_NM','STD_CLSS_NM'])['count'].sum().reset_index(name='sum')

# print(city_sum.head(50))
# Top 3 - 한식 음식점, 편의점, 슈퍼마켓  지방-주유소, 서울,인천 - 비알콜 음식점  경기 - 기타 대형 종합 소매업 

data['gap']= data['CNT'] - data['CSTMR_CNT']

data.loc[data['gap'] <0,'mark'] = '취소있음'
data.loc[data['gap'] ==0,'mark'] = '고객다름'
data.loc[data['gap'] >0,'mark'] = '단골있음'

gap=data.groupby('STD_CLSS_NM')['mark'].value_counts().reset_index(name='count')


# 카드 취소 내역이 있는 업종

# df=gap.groupby('STD_CLSS_NM')['count'].sum().reset_index().merge(gap[gap['mark']=='취소있음'][['STD_CLSS_NM','count']],on='STD_CLSS_NM')
# df.rename(columns={'count_x': 'total',
#                    'count_y': 'cancel_count'},inplace=True)
# df['cancel_rate'] = (df['cancel_count']/df['total'])*100

# df=df.sort_values('cancel_rate',ascending=False,ignore_index=True)
# print(df.head(10))

#           STD_CLSS_NM        total      cancel_count      rate
# 1       정기 항공 운송업      115914         38014        32.79%
# 2           여행사업          47446          5537        11.67%            항공, 여행 사업의 결제 취소율 높음
# 3       건강보조식품 소매업    83857          2782        3.32%           
# 4        내항 여객 운송업      72294          1857        2.57%
# 5        스포츠 용품 임대업    466533         9912        2.12% 

# 새로운 고객으로만 구성된 업종

df=gap.groupby('STD_CLSS_NM')['count'].sum().reset_index().merge(gap[gap['mark']=='고객다름'][['STD_CLSS_NM','count']],on='STD_CLSS_NM')
df.rename(columns={'count_x': 'total',
                   'count_y': 'differ_count'},inplace=True)
df['new_rate'] = (df['differ_count']/df['total'])*100

df=df.sort_values('new_rate',ascending=False,ignore_index=True)
print(df.head(10))

#                 STD_CLSS_NM                total     differ_count       rate
# 0       기타 분류안된 오락관련 서비스업       288           202          70.138889    주로 관광 관련
# 1               자동차 임대업               26024         15043        57.804334
# 2                  여행사업                 47446         24418        51.464823
# 3       관광 민예품 및 선물용품 소매업       78793         40065        50.848426
# 4                일식 음식점업             539071        247091        45.836448
# 5             건강보조식품 소매업           83857         37272        44.447094

# 단골 고객이 있는 업종

df=gap.groupby('STD_CLSS_NM')['count'].sum().reset_index().merge(gap[gap['mark']=='단골있음'][['STD_CLSS_NM','count']],on='STD_CLSS_NM')
df.rename(columns={'count_x': 'total',
                   'count_y': 'differ_count'},inplace=True)
df['regular_rate'] = (df['differ_count']/df['total'])*100

df=df.sort_values('regular_rate',ascending=False,ignore_index=True)
df.head(20)

#              STD_CLSS_NM     	   total	differ_count	    rate
# 0	          버스 운송업	       192281   	171896	       0.893983    일상생활 밀접
# 1	         체인화 편의점	       3210466	   2867733	       0.893245
# 2	           슈퍼마켓	          1630700	   1445836	      0.886635
# 3	      기타 대형 종합 소매업	   1495163	    1317200	       0.880974
# 4	            면세점	          144349	   122761	      0.850446
# 5	   기타음식료품위주종합소매업	558716	     469163	        0.839716


#데이터 원상복구 
data.drop(['gap','mark'],axis=1,inplace=True)


