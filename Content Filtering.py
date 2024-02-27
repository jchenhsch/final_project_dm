#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# In[2]:


item_df = pd.read_csv('anime.csv')
synopsis_df = pd.read_csv('anime_with_synopsis.csv')

synopsis_df.drop(['Name', 'Score', 'Genres'], axis=1, inplace=True)
synopsis_df.head()


# In[3]:


# Join on the 'id' column with a left join
anime_df = pd.merge(item_df, synopsis_df, on='MAL_ID')

print(anime_df.head())


# ## Preprocessing before vectorizing the item traits

# In[4]:


# clean studios to ensure minor spelling and punctuation differences don't cause one studio to have many unique values

def clean_text(df, col_name):
  # Define regular expression pattern for removing unwanted characters
  pattern = r"[^\w]"
  # Clean studio names using apply and lambda function
  df[col_name] = df[col_name].apply(lambda x: re.sub(pattern, "", x.lower()).strip())
  return df

# Apply cleaning function
anime_df = clean_text(anime_df, 'Studios')


# In[5]:


# Take a look at unique values lists to judge one-hot encoding possibilities
unique_ratings = anime_df['Rating'].unique()
print(len(unique_ratings))

unique_studios = anime_df['Studios'].unique()
print(len(unique_studios))

unique_genre_combos = anime_df['Genres'].unique()
print(len(unique_genre_combos))

unique_type = anime_df['Type'].unique()
print(len(unique_type))
print(anime_df['Type'].value_counts())


# In[6]:


# Reduce number of uniques for Studios
studios_counts = anime_df['Studios'].value_counts()

# for all studios that do not produce much content, lump them into 'unknown'
def modify_studios(df, threshold=15):
  rare_studios = df['Studios'].value_counts()[lambda x: x < threshold].index.tolist()
  df.loc[df['Studios'].isin(rare_studios), 'Studios'] = 'unknown'
  return df

# Apply modification function
anime_df = modify_studios(anime_df)

# Print modified DataFrame
print(anime_df['Studios'].value_counts())


# In[7]:


# Find all unique genre values for one-hot encoding later

# Function to extract unique words
def get_unique_genres(df, col_name='Genres'):
  # Split each row's genres into a list of words
  df[col_name] = df[col_name].str.split(',')
  # combine into a single list of all words
  all_genres = df[col_name].sum()
  # Get unique words
  unique_genres = list(set(genre.strip() for genre in all_genres))
  return unique_genres

unique_genres = get_unique_genres(anime_df)
print(unique_genres)



# In[8]:


# clean 'Premiered' to just show a year or 'unknown'. later create dummy variables for each decade

def extract_year(df, col_name='Premiered'):
  # Define regular expression pattern to extract year
  pattern = r"\d{4}"  # Matches 4 consecutive digits
  # Extract year using apply and lambda function
  df[col_name] = df[col_name].apply(lambda x: re.search(pattern, x).group() if re.search(pattern, x) else "unknown")
  return df

# Apply function
anime_df = extract_year(anime_df)

# Print modified DataFrame
print(anime_df['Premiered'].value_counts())


# In[9]:


# There are some animes with 'Score' unknown. If the score is unknown than there must be very little engagement
# with these titles. Punish these products by imputing a below avg score equal to decile 1

# Coerce to turn strings into NaN
anime_df['Score'] = pd.to_numeric(anime_df['Score'], errors='coerce')
# Create a mask for rows where 'Score' is not NaN
mask = anime_df['Score'].notna()
filtered_df = anime_df[mask]

# Calculate the first decile 
D1 = filtered_df['Score'].quantile(0.1)
print(D1)

# Impute
anime_df['Score'] = anime_df['Score'].fillna(D1)


# In[10]:


# Scale the numerical features 'Score' and 'Popularity'

scaler = MinMaxScaler()
# Select columns
cols_to_scale = ['Score', 'Popularity']
# Fit the scaler
scaler.fit(anime_df[cols_to_scale])
# Transform the data
anime_df[cols_to_scale] = scaler.transform(anime_df[cols_to_scale])
print(anime_df)


# ## One-hot encoding/dummy variables for categorical features

# In[11]:


# Perform one-hot encoding on the 'Type' column
type_dummies = pd.get_dummies(anime_df['Type'], prefix='Type')
# Concatenate the new columns (one-hot encoded) to the original DataFrame
anime_df = pd.concat([anime_df, type_dummies], axis=1)
# Drop the original column
anime_df.drop('Type', axis=1, inplace=True)

# Perform one-hot encoding on the 'Rating' column
type_dummies = pd.get_dummies(anime_df['Rating'], prefix='Rating')
# Concatenate the new columns (one-hot encoded) to the original DataFrame
anime_df = pd.concat([anime_df, type_dummies], axis=1)
# Drop the original column
anime_df.drop('Rating', axis=1, inplace=True)

# Perform one-hot encoding on the 'Studios' column
type_dummies = pd.get_dummies(anime_df['Studios'], prefix='Studios')
# Concatenate the new columns (one-hot encoded) to the original DataFrame
anime_df = pd.concat([anime_df, type_dummies], axis=1)
# Drop the original column
anime_df.drop('Studios', axis=1, inplace=True)

anime_df.head()


# In[12]:


# List of genres to create dummy variables for
genres_list = ['Thriller', 'Demons', 'Parody', 'Mecha', 'Historical', 'Horror', 'Shounen Ai', 'Josei', 'Dementia', 'Action', 'Romance', 'Police', 'School', 'Vampire', 'Seinen', 'Fantasy', 'Ecchi', 'Space', 'Unknown', 'Sports', 'Psychological', 'Samurai', 'Game', 'Drama', 'Sci-Fi', 'Slice of Life', 'Martial Arts', 'Music', 'Cars', 'Magic', 'Comedy', 'Mystery', 'Shoujo Ai', 'Military', 'Yaoi', 'Shoujo', 'Supernatural', 'Adventure', 'Kids', 'Super Power', 'Harem', 'Shounen']

# Create a dummy column for each genre
for genre in genres_list:
    anime_df[genre] = anime_df['Genres'].apply(lambda x: genre in x).astype(int)
    
anime_df.head()


# In[13]:


anime_df.drop('Genres', axis=1, inplace=True)


# In[14]:


# Turn decades('Premiered') into dummy variables
def year_to_decade(year):
    if year == 'unknown':
        return 'unknown'
    else:
        year = int(year)
        return f'{(year // 10) * 10}s'  # Convert to decade format like '1990s'

# Apply the function to the 'Premiered' column
anime_df['Decade'] = anime_df['Premiered'].apply(year_to_decade)
# Perform one-hot encoding on the 'Decade' column
decade_dummies = pd.get_dummies(anime_df['Decade'])
# Concatenate the new columns (one-hot encoded) to the original DataFrame
anime_df = pd.concat([anime_df, decade_dummies], axis=1)

# Drop the 'Decade' column as it's no longer needed
anime_df.drop(['Decade', 'Premiered'], axis=1, inplace=True)

anime_df.head()


# In[15]:


# Remove NaNs from 'sypnopsis'
anime_df['sypnopsis'] = anime_df['sypnopsis'].fillna('')
# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
# Apply TF-IDF to the 'Synopsis' column
tfidf_matrix = vectorizer.fit_transform(anime_df['sypnopsis'])
# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
# Concatenate this DataFrame with your main DataFrame
anime_df = pd.concat([anime_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

anime_df.drop('sypnopsis', axis=1, inplace=True)


# In[16]:


pd.set_option('display.max_columns', None)  # Set to display all columns
pd.set_option('display.max_rows', 10)       # Set to display 10 rows for brevity

# Print the DataFrame
print(anime_df.head())


# ## Create a similarity matrix and calculate cosine similarity for content-based recommendations

# In[17]:


# Create High, Medium, Low Weights to ensure similarity scores are more tied to the important features 

# Define the weights for each category
very_high_weight = 2.25
high_weight = 1.75
medium_weight = 1.5

# Lists of column names for each weight category
very_high_weight_columns = ['Score']
high_weight_columns = ['Popularity', 'Type_Movie', 'Type_Music', 'Type_ONA', 'Type_OVA', 'Type_Special', 'Type_TV', 'Type_Unknown',
                       'Rating_G - All Ages', 'Rating_PG - Children', 'Rating_PG-13 - Teens 13 or older', 
                       'Rating_R - 17+ (violence & profanity)', 'Rating_R+ - Mild Nudity', 'Rating_Unknown', 'Thriller',
                       'Demons', 'Parody', 'Mecha', 'Historical', 'Horror', 'Shounen Ai', 'Josei', 'Dementia', 'Action',
                       'Romance', 'Police', 'School', 'Vampire', 'Seinen', 'Fantasy', 'Ecchi', 'Space', 'Unknown',
                       'Sports', 'Psychological', 'Samurai', 'Game', 'Drama', 'Sci-Fi', 'Slice of Life', 'Martial Arts',
                       'Music', 'Cars', 'Magic', 'Comedy', 'Mystery', 'Shoujo Ai', 'Military', 'Yaoi', 'Shoujo',
                       'Supernatural', 'Adventure', 'Kids', 'Super Power', 'Harem', 'Shounen', '1960s','1970s','1980s',
                       '1990s','2000s','2010s', '2020s','unknown']
medium_weight_columns = ['Studios_8bit','Studios_a1pictures','Studios_acgt','Studios_actas','Studios_aic',
                         'Studios_aicasta','Studios_aicplus','Studios_ajiado','Studios_arms','Studios_artland',
                         'Studios_asahiproduction','Studios_asread','Studios_bandainamcopictures',
                         'Studios_bcmaypictures','Studios_beetrain','Studios_bones','Studios_brainsbase',
                         'Studios_cloverworks','Studios_comixwavefilms','Studios_creatorsinpack','Studios_daume',
                         'Studios_davidproduction','Studios_diomedéa','Studios_dle','Studios_dogakobo','Studios_dwarf',
                         'Studios_eiken','Studios_emtsquared','Studios_fanworks','Studios_feel','Studios_gaina',
                         'Studios_gainax','Studios_gallop','Studios_gathering','Studios_gohands','Studios_gonzo',
                         'Studios_grouptac','Studios_halfilmmaker','Studios_haolinersanimationleague',
                         'Studios_hoodsentertainment','Studios_ilca','Studios_jcstaff','Studios_kachidokistudio',
                         'Studios_kinemacitrus','Studios_kyotoanimation','Studios_lerche','Studios_lidenfilms',
                         'Studios_madhouse','Studios_magicbus','Studios_manglobe','Studios_mappa','Studios_millepensee',
                         'Studios_mushiproduction','Studios_nipponanimation','Studios_nomad','Studios_olm',
                         'Studios_onionskin','Studios_passione','Studios_paworks','Studios_polygonpictures',
                         'Studios_productionig','Studios_productionigxebec','Studios_productionims',
                         'Studios_productionreed','Studios_projectno9','Studios_radix','Studios_rganimationstudios',
                         'Studios_saigonoshudan','Studios_sanzigen','Studios_satelight','Studios_seven',
                         'Studios_sevenarcs','Studios_sevenarcspictures','Studios_shaft',
                         'Studios_shanghaifochfilmcultureinvestment','Studios_shineianimation','Studios_shuka',
                         'Studios_signalmd','Studios_silverlink','Studios_sparklykeyanimationstudio',
                         'Studios_studio4c','Studios_studiocolorido','Studios_studiocomet','Studios_studiodeen',
                         'Studios_studiofantasia','Studios_studioghibli','Studios_studiogokumi','Studios_studiohibari',
                         'Studios_studiopierrot','Studios_studiopuyukai','Studios_sunrise','Studios_synergysp',
                         'Studios_tatsunokoproduction','Studios_telecomanimationfilm','Studios_tezukaproductions',
                         'Studios_tmsentertainment','Studios_tnk','Studios_toeianimation','Studios_tokyomovieshinsha',
                         'Studios_trianglestaff','Studios_trigger','Studios_tyoanimations','Studios_ufotable',
                         'Studios_unknown','Studios_whitefox','Studios_witstudio','Studios_xebec', 
                         'Studios_yumetacompany','Studios_zerog','Studios_zexcs']

# Apply the weights
anime_df[very_high_weight_columns] *= very_high_weight
anime_df[high_weight_columns] *= high_weight
anime_df[medium_weight_columns] *= medium_weight

# Now, anime_df has weighted columns based on their importance


# In[18]:


# Separate ID and features
anime_id = anime_df['MAL_ID']
anime_name = anime_df['Name']
features = anime_df.drop(['MAL_ID', 'Name'], axis=1)
indices = pd.Series(anime_df.index, index=anime_df['Name']).drop_duplicates()

# Calculate cosine similarity
similarity_matrix = cosine_similarity(features)

# Convert to DataFrame for better readability (optional)
similarity_df = pd.DataFrame(similarity_matrix, index=anime_name, columns=anime_name)

similarity_df.head()


# In[22]:


# Function to get recommendations
def get_recommendations(title, similarity_matrix=similarity_matrix):
    # Get the index of the anime that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all animes with that anime
    sim_scores = list(enumerate(similarity_matrix[idx]))
    # Sort the animes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar animes
    sim_scores = sim_scores[1:11]
    # Get the anime indices and scores
    anime_indices_scores = [(anime_df['Name'].iloc[i[0]], i[1]) for i in sim_scores]
    # Return the top 10 most similar animes along with their scores
    return anime_indices_scores

# Get recommendations for a given anime title
recommendations = get_recommendations('Betterman')

for title, score in recommendations:
    print(f"{title}: {score:.2f}")


# In[ ]:





# In[ ]:




