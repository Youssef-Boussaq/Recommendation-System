# Import required libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# Step 1: Sample rating data
# Each entry is a user rating a specific movie
ratings_dict = {
    "user_id": [1, 1, 1, 2, 2, 3, 3, 4],
    "movie":   ["Matrix", "Titanic", "Avatar", "Matrix", "Titanic", "Titanic", "Avatar", "Matrix"],
    "rating":  [5, 4, 3, 5, 3, 2, 4, 3]
}
df = pd.DataFrame(ratings_dict)

# Step 2: Create user-item matrix
# Rows = users, Columns = movies, Values = ratings
matrix = df.pivot_table(index='user_id', columns='movie', values='rating')

# Step 3: Fill missing values (NaN) with 0 to apply SVD
matrix_filled = matrix.fillna(0)

# Step 4: Apply Truncated SVD to factorize the matrix
svd = TruncatedSVD(n_components=2)  # Use 2 latent features (you can increase this)
U = svd.fit_transform(matrix_filled)  # User-feature matrix
Sigma = svd.singular_values_         # Singular values (used internally)
VT = svd.components_                 # Movie-feature matrix

# Note: Sigma is already integrated into U and VT in scikit-learn's TruncatedSVD
# So we can directly multiply U and VT
approx_matrix = np.dot(U, VT)

# Step 5: Convert the predicted matrix back into a DataFrame
predicted_ratings = pd.DataFrame(approx_matrix, index=matrix.index, columns=matrix.columns)

# Step 6: Show predicted ratings for all users and movies
print("🔮 Predicted Ratings:")
print(predicted_ratings.round(2))


# Step 7: Function to recommend top N movies for a specific user
def get_recommendation_for_user(user_id, original_matrix, predicted_matrix, top_n=2):
    # Get movies the user has already rated
    rated_movies = original_matrix.loc[user_id].dropna().index.tolist()
    
    # Get predicted ratings for all movies for the user
    user_predictions = predicted_matrix.loc[user_id]
    
    # Exclude movies already rated
    user_predictions = user_predictions.drop(index=rated_movies)
    
    # Sort predictions by highest rating and return top N
    recommendations = user_predictions.sort_values(ascending=False).head(top_n)
    return recommendations


# Step 8: Example – Get top recommendations for user 2
recs_user2 = get_recommendation_for_user(2, matrix, predicted_ratings)
print("🎬 Recommended movies for User 2:")
print(recs_user2)
