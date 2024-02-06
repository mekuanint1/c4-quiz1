import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import seaborn as sns
from collections import Counter
data = pd.read_csv('/home/mekuanint/CHPC2024/movie_dataset.csv')
data.info()
data = data.interpolate(method='linear') # Interpolate for all the data


# ##### Q1. What is the highest rated movie in the dataset? 
R_max=data['Rating'].max() # Tha highest rate
R_max_index = data.query("Rating == 9").index[0] # Index if the highest rated movie, which is 54
print(data.loc[R_max_index, 'Title']) # The highest rated movie


# ##### Q2. What is the average revenue of all movies in the dataset? 
Rev_averagege = data['Revenue (Millions)'].mean() # The average revenue of all movies
print (Rev_averagege)


# ##### Q3. What is the average revenue of movies from 2015 to 2017 in the dataset?
index_15_17=[] # Index of the movies from 2015 to 2017
count=0
tot_rev=0
while count < len(data['Year']):
    if 2015 <= data['Year'][count] <= 2017:
        index_15_17.append(count)
        tot_rev=tot_rev + data['Revenue (Millions)'][count]
    count= count+1
Rev_average=tot_rev/len(index_15_17)
print(Rev_average)


# ##### Q4.  How many movies were released in the year 2016? 
Mov_per_year = data['Year'].value_counts() # Number of movies per year
print(Mov_per_year)


# ##### Q5. How many movies were directed by Christopher Nolan? 
Movies_Chris = data.pivot_table(index = ['Director'], aggfunc = "size") # movies were directed by Christopher Nolan
print(Movies_Chris['Christopher Nolan'])


# ##### Q6. How many movies in the dataset have a rating of at least 8.0?
#Rat_fre = data.pivot_table(index = ['Rating'], aggfunc = "size")
Rat_fre = data['Rating'].value_counts()
Rat_key=Rat_fre.keys()
R_g8=[]
for i in Rat_key:
    if i >= 8:
       R_g8.append(Rat_fre[i])
np.sum(R_g8)        


# ##### Q7. What is the median rating of movies directed by Christopher Nolan? 
Movie_direct=data['Director']
b = (data[data['Director'] =='Christopher Nolan']).index.tolist()
rat_chri=[]
for i in b:
    rat_chri.append(data['Rating'][i])
rat_chri
np.median(rat_chri)


# ##### Q8. Find the year with the highest average rating? 
years = data['Year'].value_counts()
year_avg_score = data.loc[data['Year'].isin(years.index)].groupby('Year')['Rating'].mean()
print(year_avg_score)


# ##### Q9. What is the percentage increase in number of movies made between 2006 and 2016? 
Mov_2006 = 44   # Number of movies released in 2006
Mov_2016 = 297  # Number of movies released in 2016
per_inc= (Mov_2016 - Mov_2006)*100/(Mov_2006) # percentage increase
print(per_inc)


# ##### Q10. Find the most common actor in all the movies?
Act_rep = '[{}]'.format(','.join(map(str, data['Actors'])))
resa=Act_rep.replace(", ", ",")
c=Counter(resa.split(","))
for name, freq in c.items():
    if freq == max(c.values()):
        print(name)


# ##### Q11. How many unique genres are there in the dataset?
All_genre=[]
for i in range(0,len(data['Genre'])):
      All_genre.append(data['Genre'][i].split(","))
count_g=0
concat_dg=[]
while count_g < len(All_genre):
    dfg=All_genre[count_g]
    concat_dg= concat_dg+dfg
    count_g=count_g+1
cg = Counter(concat_dg) 
print(len(cg.keys()))


# ##### Q12. Do a correlation of the numerical features, what insights can you deduce? Mention at least 5 insights.

data_for_corr = data.drop(['Rank', 'Title', 'Genre', 'Description', 'Actors', 'Director'], axis=1)
corr = data_for_corr.corr(method = 'pearson')
print(corr)
plt.figure(figsize=(26,26))
sns.heatmap(corr, annot=True, xticklabels=True, yticklabels=True, cmap='plasma', annot_kws={'size': 20})
plt.title("Correlation of the numerical features", size=26)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

