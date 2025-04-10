# Recommendation-System

🎬 Movie Recommendation System using SVD (Singular Value Decomposition)
 Description:
This project demonstrates how to build a simple movie recommendation system using linear algebra techniques, specifically SVD (Singular Value Decomposition). The system predicts missing user ratings for movies and recommends movies based on these predictions.

✅ What It Does:
Takes user-movie rating data as input.

Builds a user-item matrix, where:

Rows = users

Columns = movies

Values = user ratings

Fills missing ratings with zeros to prepare for matrix decomposition.

Applies SVD to decompose the matrix into lower-dimensional representations:

User features

Movie features

Reconstructs an approximate matrix with predicted ratings for all movies, even those not yet rated.

Provides personalized recommendations by:

Identifying which movies a user hasn’t rated

Predicting their ratings

Suggesting the top N movies with the highest predicted ratings

📊 Techniques Used:
Linear Algebra: Matrix decomposition via SVD

Data Preprocessing: Handling missing values

Matrix Factorization: Extracting hidden patterns between users and movies

Python Libraries:

numpy for numerical computation

pandas for data manipulation

scikit-learn for Truncated SVD

👨‍💻 Example Use Case:
User 2 has watched "Matrix" and "Titanic". The system uses SVD to predict that they might also like "Avatar" based on patterns in other users’ ratings. So "Avatar" will be recommended to them.