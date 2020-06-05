

import pandas as pd
import matplotlib.pyplot as plt


wine = pd.read_csv("./data/winequality-white.csv", sep = ';', header = 0)


coundt_data = wine.groupby('quality')['quality'].count() # quailty에 있는 각 개체 별로 갯수를 세겠다, 여기서는 3456789

'''
print(coundt_data)

quality
3      20
4     163
5    1457
6    2198     -> 5,6에 수렴 
7     880
8     175
9       5

coundt_data.plot()
plt.show()

'''


