
# Pandas

# 1차원 배열 표 -> Series (라벨이 붙은 1차원 데이터)
# 2차원 배열 표 -> Dataframe (여러 series가 모인 데이터)

# int64 -> 64bit의 크기를가진 정수 2^-63 ~ 2^63까지 정수를 처리할 수 있음, in32, b00l ( 0 or 1) 등등이 있다


#233p  데이터와 인덱스 추출

# Seris 자료형 -> series.value()로 데이터값 추출, series.index로 인덱스 추출


#236p

# series = series.drop("strawberry") 
#  strawberry 삭제


#237p 필터링 

# series = series[series >= 5][series < 10] 
# 조건이 여러개일 때 [] [] 을 연속해서 덧붙이면 된다. 

#238p 정렬

# series 인덱스 정렬 -> series.sort_index()
# series 데이터 정렬 -> series.sort_values()

# default = ascending = True ( 오름차순 )


# Dataframe형 변수 df의 인덱스는 df.index에 행 수와 같은 길이의 리스트를 대입하며 설정할 수 있다. 

#                 df의 컬럼은 df.columns에 열 수와 같은 길이의 리스트를 대입하여 설정할 수 있다.


# df.append( series3, ignore_index = True) => 행 추가
# df["new"] = new_column                   => 열 추가

# 데이터 참조
# loc -> 이름으로 참조
# iloc -> 번호로 참조

# df.drop()  -> 행 또는 열 삭제

# df.sort_values( by = "column or column list", ascending = True) -> 지정한 열의 값을 오름차순 정렬

# df = df.loc[df["apple"] >= 5]   -> df의 "apple" 열이 5 이상 값을 가진 행을 포함한 Dateframe을 df에 대입

