
import pandas as pd
from math import sqrt
import operator
from sklearn.cluster import KMeans


csv_file = 'C:/Users/onkar/Downloads/AMS 595/team_project/ml-latest-small/ratings.csv'

filename = 'C:/Users/onkar/Downloads/AMS 595/team_project/user_movieid.csv'


## reading data into a dict of dicts
data_dict = {}
with open(csv_file, 'r') as f:
    next(f)
    for items in f:
        temp = temp = [float(value) for value in items.split('\n')[0].split(',')]
        if temp[0] in data_dict:
            data_dict[temp[0]][temp[1]] = [temp[2],temp[-1]]
        else:
            data_dict[temp[0]] = {}
            data_dict[temp[0]][temp[1]] = [temp[2],temp[-1]]
            




def eucledian_distance(var_data,var_user1,var_dict):
    eu_dist = 0
    union = var_data[var_user1].keys() | var_dict.keys()
    intersect = var_data[var_user1].keys() & var_dict.keys()
    for item in union:
        if item in intersect:
            eu_dist+=(var_data[var_user1][item][0] - var_dict[item])**2
        elif item in var_data[var_user1].keys():
            eu_dist+=var_data[var_user1][item][0]**2
        else:
            eu_dist+=var_dict[item]**2
    return eu_dist**0.5



##denominator of cosine
def modulus(var_data):
    temp = 0
    for item in var_data.keys():
        temp += var_data[item][0]*var_data[item][0]
    return sqrt(temp)



##cosine similarity
def cosine_similarity(var_data, var_user_dict):
    modb = modulus(var_user_dict)
    sim_list = []
    for item in var_data.keys():
        #if item == var_user:
         #   pass
        #else:
        moda = modulus(var_data[item])
        intersect = var_data[item].keys() & var_user_dict.keys()
        temp = 0
        for movie in intersect:
            temp+= var_data[item][movie][0]*var_user_dict[movie][0]
            
        sim_list.append(temp/(moda*modb))
    return sim_list
            

##recommend movies via weighted average
def movie_names(var_data,var_sim,var_user_dict,var_n):
    movie_dict = {}
    sim_dict = {}
    similar_user = sorted(var_sim,reverse=True)[:var_n]
    for i in range(0,var_n):
        most_similar_user = var_sim.index(similar_user[i])+1
                
        intersect = data_dict[most_similar_user].keys() & var_user_dict.keys()
        for item in data_dict[most_similar_user].keys():
            if item not in intersect:
                if item in movie_dict:
                    movie_dict[item] += data_dict[most_similar_user][item][0]*similar_user[i]
                    sim_dict[item] += similar_user[i]
                else:
                    movie_dict[item] = data_dict[most_similar_user][item][0]*similar_user[i]
                    sim_dict[item] = similar_user[i]
        
        
    rank_dict = dict(zip(movie_dict.keys(),[0]*len(movie_dict.keys())))
    for item in movie_dict.keys():
        rank_dict[item] = movie_dict[item]/sim_dict[item]
    return sorted(rank_dict.items(),key=operator.itemgetter(1),reverse=True)[:var_n]
        
    


##basic run of algo
sim_arr = cosine_similarity(data_dict,data_dict[5])
recommended_movies = movie_names(data_dict,sim_arr,data_dict[5],10)
print(recommended_movies)

### k means 

#reading from the csv created from the dict
data_df = pd.read_csv(filename)
data_df = data_df.fillna(0)
del data_df['Unnamed: 0']


def cluster_kmeans(var_data_df, var_user, var_cluster=10,var_recommend = 10):

    kmeans = KMeans(n_clusters=var_cluster, random_state=0).fit(var_data_df)
    var_data_df['cluster_id'] = kmeans.labels_

    data_subset = var_data_df.loc[var_data_df['cluster_id'] == var_data_df.loc[var_user]['cluster_id']]

    for cols in data_subset.columns[:-2]:
        if data_subset.loc[5][cols] > 0:
            del data_subset[cols]
    
    sorted_series = data_subset.sum(axis=1).sort_values(ascending=False)
    
    return list(sorted_series.index[:var_recommend])



cluster_kmeans(data_df,5)

##Hybrid model


def hybrid_model(var_data_df, var_data_dict, var_user, var_clusters = 5, var_recommendations = 10):

    kmeans = KMeans(n_clusters=var_clusters, random_state=0).fit(var_data_df)
    var_data_df['cluster_id'] = kmeans.labels_
    data_subset = var_data_df.loc[var_data_df['cluster_id'] == var_data_df.loc[var_user]['cluster_id']]
    subset_dict = { item+1: var_data_dict[item+1] for item in list(data_subset.index)}

    sim_arr = cosine_similarity(subset_dict,subset_dict[var_user])
    recommended_movies = movie_names(subset_dict,sim_arr,subset_dict[var_user],var_recommendations)
    return recommended_movies



print(hybrid_model(data_df,data_dict,5))
