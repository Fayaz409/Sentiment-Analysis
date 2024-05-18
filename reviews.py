from flask import Flask, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from model import SentimentAnalysisModel
import os
import pandas as pd
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user_reviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
# Assuming paths are set correctly to where the model and its components are saved
model_path = r'C:\Users\ftuni\OneDrive - National University of Sciences & Technology\Documents\Sentiment_Model/model.h5'
vectorizer_path = r'C:\Users\ftuni\Desktop\AnnotateFlask\Sentiment_Model\vectorizer.pkl'
encoder_path = r'C:\Users\ftuni\Desktop\AnnotateFlask\Sentiment_Model\encoder.pkl'
global model  # Declare the model as a global variable if needed

from model import SentimentAnalysisModel  # Ensure this import does not reload the model itself

def load_model():
    """Function to load and return the model."""
    global model
    model = SentimentAnalysisModel(model_path, vectorizer_path, encoder_path, l2_reg=0.01)
    return model

model = load_model()
class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), nullable=False)
    product_name = db.Column(db.String(100), nullable=False)
    review_content = db.Column(db.Text, nullable=False)
    review_time = db.Column(db.DateTime, default=func.now())
    sentiment = db.Column(db.String(50), nullable=True)  # Store the sentiment

    def __repr__(self):
        return f"<Review {self.username} - {self.product_name}>"

@app.route('/')
def index():
    # Read the CSV file
    df = pd.read_csv(r'C:\Users\ftuni\Desktop\AnnotateFlask\amazon.csv')
    # Filter out rows where the image link is not available or is null
    df = df[df['img_link'].notna() & df['img_link'].str.strip().astype(bool)]
    # Create a list of dictionaries for each product with necessary details
    products = df[['product_name', 'discounted_price', 'img_link']].to_dict(orient='records')
    return render_template('index.html', products=products)
@app.route('/submit_review', methods=['POST'])
def submit_review():
    email = request.form['email']
    username = request.form['username']
    product_name = request.form['product_name']
    review_content = request.form['review']
    sentiment = model.predict_texts([review_content])[0]  # Assuming predict_texts returns a list
    print(sentiment)
    if sentiment==0:
        sentiment='Positive'
    else:
        sentiment='Negative'

    # Save all reviews with sentiment
    review = Review(email=email, username=username, product_name=product_name,
                    review_content=review_content, sentiment=sentiment)
    db.session.add(review)
    db.session.commit()
    return render_template('thanks.html' , email=email, username=username)
    

@app.route('/reviews', methods=['GET'])
def reviews():
    reviews = Review.query.all()
    return render_template('all_reviews.html', reviews=reviews)
@app.route('/negative_reviews', methods=['GET'])
def negative_reviews():
    reviews = Review.query.filter_by(sentiment='Negative').all()
    return render_template('negative_reviews.html', reviews=reviews)

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('reviews.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)
