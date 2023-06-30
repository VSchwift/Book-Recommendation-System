# Book Recommendation System

This is a book recommendation system that uses collaborative filtering to provide personalized book recommendations based on user ratings. The system is built using Python and Streamlit framework for the web application. [Test the App here](https://book-recommendation-system-swv01.streamlit.app/) but does not work well on Streamlit. It will work fine locally.

## Overview

The book recommendation system leverages collaborative filtering, a popular technique in recommender systems, to suggest books to users based on their similarity to other users. It uses a K-Nearest Neighbors (KNN) algorithm to find the nearest neighbors to a given book based on user ratings. The system then recommends books that are highly rated by those nearest neighbors.

## Features

- Users can enter the title of a book to get personalized recommendations.
- The system displays a list of books for users to choose from.
- The recommendations are shown in a tabular format, including the book title and a distance metric representing the closeness of the recommendation to the input book.
- The system provides an intuitive and interactive user interface using Streamlit.

## Data
The system uses preprocessed data files containing book information and user ratings. These data files are stored in Feather format for efficient read/write operations. The data files can be obtained from [source location](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) and should be placed in the data directory.

## Model
The K-Nearest Neighbors (KNN) model used for collaborative filtering is trained and saved using the joblib library. The trained model file (KNN_model.joblib) should be placed in the models directory.

## How to Use
```bash
pip install -r requirements.txt
streamlit run app.py
