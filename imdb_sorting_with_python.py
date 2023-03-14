############################################
# IMDB Movie Scoring & Sorting
############################################

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.stats as st

# pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_ = pd.read_csv('datasets/movies_metadata.csv')
df = df_.copy()

df = df[['title', 'vote_average', 'vote_count']]

df.head()
df.shape
df.info()
df.isnull().sum()
df.dropna(inplace=True)
df.describe().T

############################################
# Sorting by Vote Count or Average
############################################

df.sort_values('vote_average', ascending=False)

df['vote_count'].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T

df[df['vote_count'] > 400].sort_values('vote_average', ascending=False)

df['vote_count_score'] = MinMaxScaler(feature_range=(1, 10)).fit(df[['vote_count']]).transform(df[['vote_count']])

df.sort_values('vote_count_score', ascending=False)

############################################
# Sorting by Vote Count and Average
############################################

df['average_count_score'] = df['vote_average'] * df['vote_count_score']

df.sort_values('average_count_score', ascending=False).head(10)

############################################
# IMDB Weighted Rating
############################################

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report

# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

M = 2500
C = df['vote_average'].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)

df.sort_values('average_count_score', ascending=False)

weighted_rating(7.4, 11444.00, M, C) # Deadpool

weighted_rating(8.10, 14075.00, M, C) # Inception

weighted_rating(8.50000, 8358.00000, M, C) # The Shawshank Redemption

df['weighted_rating'] = weighted_rating(df["vote_average"], df["vote_count"], M, C)
# df['weighted_rating2'] = df.apply(lambda x: weighted_rating(x["vote_average"], x["vote_count"] , M, C), axis=1)

df.head()

df.sort_values('weighted_rating', ascending=False).head(10)

############################################
# Sorting by Bayesian Average Rating Score
############################################

df2_ = pd.read_csv('datasets/imdb_ratings.csv')
df2 = df2_.copy()

df2.drop('Unnamed: 0', axis=1, inplace=True)
df2.iloc[0:, 1:]

#                                     movieName  rating      ten    nine   eight   seven    six   five   four  three   two    one
# 0    1.       The Shawshank Redemption (1994)    9.20  1295382  600284  273091   87368  26184  13515   6561   4704  4355  34733
# 1               2.       The Godfather (1972)    9.10   837932  402527  199440   78541  30016  16603   8419   6268  5879  37128
# 2      3.       The Godfather: Part II (1974)    9.00   486356  324905  175507   70847  26349  12657   6210   4347  3892  20469
# 3             4.       The Dark Knight (2008)    9.00  1034863  649123  354610  137748  49483  23237  11429   8082  7173  30345
# 4                5.       12 Angry Men (1957)    8.90   246765  225437  133998   48341  15773   6278   2866   1723  1478   8318

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351]) # The Shawshank Redemption (1994)
bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905]) # The Godfather (1972)

df2['bar_score'] = df2.apply(lambda x: bayesian_average_rating(x[['one', 'two', 'three', 'four', 'five',
                                                                'six', 'seven', 'eight', 'nine', 'ten']]), axis=1)

bayesian_average_rating([34733, 4355, 4704, 6561])

df2.sort_values('bar_score', ascending=False).head(10)
