# Sentiment Analysis Web Application

This is a simple web application for sentiment analysis of text reviews. It uses a neural network model trained to classify reviews as positive or negative. The application is built using Flask for the web interface and TensorFlow/Keras for the machine learning model.

## Features

* Preprocesses text by tokenizing, removing stopwords, and lemmatizing.
* Uses TF-IDF vectorization for feature extraction.
* Implements a neural network model with L2 regularization and dropout for sentiment analysis.
* Provides a simple web interface to submit text reviews and get sentiment predictions.

## Installation

### Prerequisites

* Python 3.7 or higher
* Virtual environment (recommended)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/Fayaz409/Sentiment-Analysis.git
cd Sentiment-Analysis
```

2. Create a virtual environment:

```bash
python -m venv sentana
```

3. Activate the virtual environment:

**Windows:**

```bash
sentana\Scripts\activate
```

**macOS and Linux:**

```bash
source sentana/bin/activate
```

4. Install the required packages:

```bash
pip install -r requirements.txt
```

5. Download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Running the Application

To start the web application, simply run the `reviews.py` file:

```bash
python reviews.py
```

## Project Structure

* `model.py`: Contains the `SentimentAnalysisModel` class for building, training, and using the sentiment analysis model.
* `reviews.py`: Main application script that sets up the Flask web server and handles HTTP requests.
* `requirements.txt`: List of required Python packages.

## Usage

Enter a text review in the input box on the web interface and submit. The model will analyze the sentiment of the review and display the result as either "Positive" or "Negative".

## Notes

* Make sure to train the model and save the model, vectorizer, and encoder before running the application. You can add a separate training script or include training code in `model.py`.
* Ensure the paths to the model, vectorizer, and encoder files are correctly specified in `model.py`.

## License

This project is licensed under the MIT License.

## Acknowledgments

* This project uses TensorFlow/Keras for building the neural network model.
* NLTK is used for text preprocessing.
* Flask is used for building the web application.

# The Trained Model file.


## You can find the code where I have trained my Sentiment Model which is used in the website.

The code utilizes an Amazon product review dataset. Here's a brief overview of the data preparation steps:

1. **Extracting Reviews:** Reviews are extracted from a compressed archive (.tar.gz) and saved in a designated directory.
2. **Parsing Reviews:** A regular expression is used to parse the review data from individual files and store them in a pandas DataFrame.
3. **Preprocessing Text:** Text preprocessing techniques like tokenization, stop word removal, and lemmatization are applied to clean and normalize the review text.
4. **Label Encoding:** Sentiment labels (positive or negative) are converted to numerical values using LabelEncoder from scikit-learn.
5. **Balancing Dataset:** The code addresses class imbalance by upsampling the minority class (negative reviews) to match the size of the majority class (positive reviews).
6. **TF-IDF Vectorizer:** TF-IDF vectorizer is used to transform text data into numerical features, capturing the importance of words based on their frequency in the document and the entire corpus.

**Model Building**

The sentiment analysis model is a basic neural network architecture built with TensorFlow's Keras API:

* **Input Layer:** Receives the TF-IDF vectorized representation of the review text.
* **Dense Layers:** Two dense layers with ReLU activation are used for feature extraction and non-linearity. Dropout layers are added for regularization to prevent overfitting.
* **Output Layer:** A dense layer with softmax activation outputs the probability distribution for the two sentiment classes (positive or negative).

**Training and Evaluation**

* **Train-Test Split:** The data is split into training, validation, and testing sets for model training and evaluation.
* **Model Training:** The model is trained on the training data with Adam optimizer and categorical cross-entropy loss function.
* **Model Evaluation:** The model's performance is evaluated on the validation and testing sets using metrics like accuracy, confusion matrix, and classification report.
* **L2 Regularization:** The code implements L2 regularization as a hyperparameter to control model complexity and potentially reduce overfitting.

**Saving and Loading the Model**

The trained model, TF-IDF vectorizer, and label encoder are saved using pickle for persistence and can be loaded for future predictions on new data.

**Key Features**

* **Comprehensive Text Preprocessing:** Includes tokenization, stop word removal, and lemmatization for robust text cleaning.
* **Class Imbalance Handling:** Addresses class imbalance by upsampling the minority class for better model performance.
* **L2 Regularization Hyperparameter Tuning:** Grid search is used to find the optimal L2 regularization value to improve model generalization.
* **Confusion Matrix and Classification Report:** Provides detailed insights into model performance for both classes.
* **Top TF-IDF Terms:** Identifies the most informative terms based on their TF-IDF scores in the training set.
* **Saved Model and Serialization:** Allows for easy deployment and reuse of the trained model for sentiment analysis on new data.

**How to Use This Code**

1. Install the required libraries (refer to the Requirements section).
2. Download the Amazon product review dataset.
3. Replace the data path in the code with the location of your downloaded dataset.
4. Run the script. The code will perform data preparation, model training, evaluation, and save the trained model and artifacts.
5. To use the saved model for sentiment prediction on new text data, load the model, vectorizer, and encoder, preprocess the new text data, and use the model's predict method to get the sentiment classification.

**Further Exploration**

* Experiment with different deep learning architectures (e.g., LSTMs, CNNs) for potentially better performance.
* Explore more advanced text preprocessing techniques like stemming, part-of-speech tagging, and named entity recognition.
* Utilize pre-trained word embeddings like Word2Vec or GloVe to capture semantic relationships between words.
* Integrate the sentiment analysis model into a web application or other systems for real-world use cases.

## Contact

For any questions or suggestions, please open an issue or contact [ftunio06@gmail.com](mailto:ftunio06@gmail.com).
