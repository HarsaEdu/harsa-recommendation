from flask import Flask, render_template, request, jsonify
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from dotenv import load_dotenv
import os
import pymysql
from apscheduler.schedulers.background import BackgroundScheduler
import time

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Get MySQL connection details from environment variables
mysql_host = os.getenv("MYSQL_HOST")
mysql_user = os.getenv("MYSQL_USER")
mysql_password = os.getenv("MYSQL_PASSWORD")
mysql_database = os.getenv("MYSQL_DATABASE")

# Connect to MySQL
connection = pymysql.connect(
    host=mysql_host,
    user=mysql_user,
    password=mysql_password,
    database=mysql_database
)

# Load data from MySQL tables into Pandas DataFrames
courses = pd.read_sql("SELECT * FROM courses WHERE deleted_at IS NULL", connection)
feedbacks = pd.read_sql("SELECT * FROM feedbacks WHERE deleted_at IS NULL", connection)
user_interests = pd.read_sql("SELECT * FROM user_interests", connection)

datas = """
SELECT feedbacks.id,
       feedbacks.user_id,
       feedbacks.course_id,
       feedbacks.rating,
       feedbacks.created_at AS feedback_created_at,
       courses.title,
       courses.description,
       courses.category_id,
       courses.image_url,
       CONCAT(user_profiles.first_name, ' ', user_profiles.last_name) AS instructor_name,
       courses.created_at AS course_created_at
FROM feedbacks
JOIN courses ON feedbacks.course_id = courses.id
JOIN user_profiles ON courses.user_id = user_profiles.id
WHERE feedbacks.deleted_at IS NULL;
"""

datas = pd.read_sql(datas, connection)

# Create a Reader object
reader = Reader(rating_scale=(1, 5))

# Load the data into a Surprise Dataset
data_surprise = Dataset.load_from_df(feedbacks[['user_id', 'course_id', 'rating']], reader)

# Use the KNNBasic collaborative filtering algorithm
sim_options = {'name': 'cosine', 'user_based': False}
recommendation_model = KNNBasic(sim_options=sim_options)

# Train the model on the entire dataset
trainset = data_surprise.build_full_trainset()
recommendation_model.fit(trainset)

def get_top_recommendations(model, user_id, data, user_interests, n=10):
    # Get user's interests (category_id)
    user_interest_categories = set(user_interests[user_interests['profile_id'] == user_id]['category_id'])

    # Get item predictions for the customer
    item_predictions = []
    user_rated_courses = set(data[data['user_id'] == user_id]['course_id'])

    for item_id in set(data['course_id']):
        # Skip courses the user has already rated
        if item_id in user_rated_courses:
            continue

        # Determine if the course is in the user's interest categories
        course_category = data[data['course_id'] == item_id]['category_id'].values[0]
        is_in_interest_categories = course_category in user_interest_categories

        # Predict rating and store item prediction
        predicted_rating = model.predict(user_id, item_id).est
        course_title = data[data['course_id'] == item_id]['title'].values[0]
        course_description = data[data['course_id'] == item_id]['description'].values[0]
        image_url = data[data['course_id'] == item_id]['image_url'].values[0]
        instructor_name = data[data['course_id'] == item_id]['instructor_name'].values[0]

        item_predictions.append({
            'course_id': item_id,
            'course_title': course_title,
            'course_description': course_description,
            'image_url': image_url,
            'instructor_name': instructor_name,
            'predicted_rating': predicted_rating,
            'is_in_interest_categories': is_in_interest_categories
        })

    # Sort predictions based on predicted rating and interest category status
    def sort_function(item_prediction):
        is_in_interest_categories = item_prediction['is_in_interest_categories']
        predicted_rating = item_prediction['predicted_rating']

        # Assign weights based on interest category status
        interest_category_weight = 2 if is_in_interest_categories else 1

        # Calculate weighted rating
        weighted_rating = interest_category_weight * predicted_rating

        # Sort based on weighted rating
        return weighted_rating

    item_predictions.sort(key=sort_function, reverse=True)

    # Get top recommended items
    top_recommendations = item_predictions[:n]

    return top_recommendations

def restart():
    os.system("pkill -f 'python app.py'")

# Schedule restart every 1 hour
scheduler = BackgroundScheduler()
scheduler.add_job(restart, 'interval', seconds=60 * 60 * 1)
scheduler.start()

# This is blocking the application from exiting
try:
    while True:
        time.sleep(2)
except (KeyboardInterrupt, SystemExit):
    # Not strictly necessary if daemonic mode is enabled but should be done if possible
    scheduler.shutdown()

@app.route('/')
def documentation():
    return render_template('index.html')

@app.route('/recommends', methods=['POST'])
def recommends():
    user_id = int(request.json['user_id'])
    max_recommendation = int(request.json['max'])

    # Use get_top_recommendations function to get recommendations
    top_recommendations = get_top_recommendations(recommendation_model, user_id, datas, user_interests, max_recommendation)

    return jsonify({'user_id': user_id, 'recommendations': top_recommendations})

if __name__ == '__main__':
    app.run(debug=True)