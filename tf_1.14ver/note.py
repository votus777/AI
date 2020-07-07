
'''
conda create -n tf114 python=3.6.5 anaconda


tf114 환경에 python 3.6.5 버전 설치 


설치후 

conda activate tf114 

python 

3.6.5 버전 설치 확인 

VScode 좌측 하단 버전 선택 




base 환경 접속은 activte base
나가려면 deactivate 

# cmd창 뒤로 가려면 ctrl + z 


python -m pip install --upgrade pip 으로 pip 버전 업그레이드 

그런데 다시 pip install tensorflow==1.14 하면 


ERROR: Cannot uninstall 'wrapt'  에러 뜬다 

conda remove wrapt 로 삭제해주고 설치 



그런데 다음에는 numpy 오류가 나온다 

base 버전에 맞춰서 재설치 

1.17 이상이면 이상한 FutureWarning들이 잔뜩 뜨는데 1.16 아래로 설치하면 사라진다 
tensorflow 1 버전과 numpy 최신버전이 안맞아서 뜨는 오류인듯 

'''

import tensorflow as tf
import numpy as np


print(tf.__version__)  # 1.14.0
