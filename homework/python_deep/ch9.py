
# Pandas 응용


# 인덱스나 칼럼이 일치하는 Dataframe 간의 연결

# pandas.concat ("dataFrame 리스트", axis = 0)
# 세로 방향으로 연결 될 때에는 동일한 컬럼
# 가로 방향으로 연결 될 때에는 동일한 인덱스로 연결된다. 

# 만약 인덱스나 컬럼이 일치하지 않는다면 Nan셀 생성된다. 


# pandas.concat ("dataFrame 리스트", axis = 0, keys["X","Y"])
# Keys 를 추가해 라벨 지정 -> 중복 피하기 
# df["X", "apple"] 로 "X" 컬럼 안의 "apple" 컬럼 참조 가능


# Dataframe 결합 (merge)

# 내부 결합 -> 일치하지 않는 행 삭제
# pandas.merge (df1, df2,on=Key가 될 컬럼, how= "inner")



# 외부 결합 -> 일치하지 않아도 남고, NaN 셀 생성
# pandas.merge (df1, df2,on=Key가 될 컬럼, how= "outer")
# key가 아니면서 이름이 같은 열은 접미사 _x, _y 가 붙는다 


# 이름이 다른 컬럼을 결합하기
# pandas.merge (df1, df2, left_on = "왼쪽 df의 컬럼", right_on = "오른쪽 df의 컬럼", how = "결합방식" )

# 인덱스를 key로 결합하기
# pandas.merge (df1, df2, left_index = True, right_index= True, how = "결합방식" )


# df.head(3) -> df의 첫 3행
# df.tail(3) -> df의 끝 3행

# df * 2 // df* df = df**2 (제곱) // np.sqrt(df)


# df의 통게 정보 중 "mean", "max", "min"을 꺼내서 df_des에 대입
# df_des = df.describe().loc[["mean", "max", 'min"]]


# Dataframe의 행간,열간 차이 구하기
# df.diff("행 간격 or 열 간격", axis ="0 or 1 ")

# df.diff = df.diff(-2, axis=0)
# df의 각 행에 대해 2행 뒤와의 차이를 계산한 Dataframe을 df_diff에 대입


# 그룹화

# df.groupby("컬럼") -> GroupBy 객체 반환 
# GroupBy 결과를 보려면 mean(), sum()등 통게함수 사용 