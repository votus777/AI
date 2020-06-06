from collections import Counter
from matplotlib import pyplot as plt  


grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]
    
    # 점수는 10점 단위로 그룹화 한다. 100점은 90점대에 속한다. 
    
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)


plt.bar([x - 4 for x in histogram.keys()], # 각 막대를 오른쪽으로 5만큼 옮기고
            histogram.values(),  10,              # 각 막대의 높이를 정해 주고                                        # give each bar a width of 8
            edgecolor = (0,0,0))                   # 너비는 10으로 하자
                                                # 각 막대의 테두리는 검은색

                                               
plt.xticks([10 * i for i in range(11)])    
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()
