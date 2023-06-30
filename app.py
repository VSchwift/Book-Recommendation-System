import numpy as np
import streamlit as st
import pandas as pd
import joblib


@st.cache_data
def load_data():
    book_df = pd.read_feather('final_df.feather')
    book_df.drop('index', axis=1, inplace=True)
    user_rating_pivot = pd.read_feather('user_rating_pivot.feather')
    user_rating_pivot.set_index('bookTitle', inplace=True)

    return book_df, user_rating_pivot

# Load the data
book_df, user_rating_pivot = load_data()

model = joblib.load('KNN_model.joblib')

st.title('Book Recommendation System')
text_input = st.text_input('Please enter the name of the book you would like recommendation to below 📚📖📔🔢')

def recommend():
    input_value = text_input.strip() 
    book_name = input_value

    a = user_rating_pivot.index.get_loc(book_name)

    distances, indices = model.kneighbors(user_rating_pivot.iloc[a,:].values.reshape(1, -1), n_neighbors = 11)
    
    recommendations = []
    for i in range(1, len(distances.flatten())):
        recommendation = (user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i])
        recommendations.append(recommendation)

    # Create a DataFrame from the recommendations list
    df_recommendations = pd.DataFrame(recommendations, columns=['Book Name', 'Distance'])
    return df_recommendations

# Initialize an empty DataFrame
df_recommendations = pd.DataFrame()

if st.button('Get Recommendations'): 
    df_recommendations = recommend()

st.write('Try this as an input➡ How To Win Friends And Influence People')
st.write('Recommendations table will have a distance column. Distance is the closeness of recommendation to the book you entered.')
show_books = pd.DataFrame(book_df['bookTitle'].unique(), columns=['Book Title'])

# Create a container with two parts
left_column, right_column = st.columns(2)

# First part of the container (left column)
with left_column:
    # Add content to the left column
    st.header('Book List')
    st.write('You can use the list below to enter a book name')
    show_books = pd.DataFrame(book_df['bookTitle'].unique(), columns=['Book Title'])
    st.dataframe(show_books)

# Second part of the container (right column)
with right_column:
    # Add content to the right column
    st.header('Recommendations')
    st.write('Your recommendations below')
    st.dataframe(df_recommendations)