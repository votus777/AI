from matplotlib import pyplot as plt  

movies = ["Annie HAll", "Ben-Hur", "Casablanca", "Gandhi", "West side stroy"]
num_oscars = [5,11,8,3,10]

#  막대의  X 좌표는 [0,1,2,3,4]. 높이는  [num_oscars]

plt.bar(range(len(movies)), num_oscars) 
plt.title("My favorite Movies")
plt.ylabel("# of Academy Awards")

plt.xticks(range(len(movies)), movies)

plt.show()

