import sys
print(sys.path)

'''
['c:\\Users\\bitcamp\\Documents\\GitHub\\AI_study\\Python\\module',   <- 현재 작업 파일 


sys에 걸려있는 path 
'C:\\Users\\bitcamp\\anaconda3\\python37.zip', 
'C:\\Users\\bitcamp\\anaconda3\\DLLs', 
'C:\\Users\\bitcamp\\anaconda3\\lib', 
'C:\\Users\\bitcamp\\anaconda3',                                   <- 제일 만만한 여기다가 집어넣자 
'C:\\Users\\bitcamp\\anaconda3\\lib\\site-packages', 
'C:\\Users\\bitcamp\\anaconda3\\lib\\site-packages\\win32', 
'C:\\Users\\bitcamp\\anaconda3\\lib\\site-packages\\win32\\lib', 
'C:\\Users\\bitcamp\\anaconda3\\lib\\site-packages\\Pythonwin']


'''


from test_import import p62_import 

p62_import.sum2() 

# 이 import는 아나콘다 폴더 C:\Users\bitcamp\anaconda3 에 들어있다
# 작업그룹 임포트 와구와구


from test_import.p62_import import sum2

sum2() 

# 작업그룹 임포트 와구와구

