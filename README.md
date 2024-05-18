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

## Contact

For any questions or suggestions, please open an issue or contact [ftunio06@gmail.com](mailto:ftunio06@gmail.com).

