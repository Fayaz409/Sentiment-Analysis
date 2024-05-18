import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SentimentAnalysisModel:
    """
    A class to build, train, evaluate, and use a sentiment analysis model.

    Methods:
    - __init__(self, l2_reg=None): Initialize the model with optional L2 regularization.
    - build_model(self): Build a neural network model with optional L2 regularization.
    - train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128): Train the model.
    - plot_accuracy(self): Plot the training and validation accuracy over epochs.
    - save_model(self, model_path, vectorizer_path, encoder_path): Save the model, vectorizer, and encoder.
    - load_model(self, model_path, vectorizer_path, encoder_path): Load a trained model, vectorizer, and encoder.
    - predict_texts(self, texts): Predict the sentiment of a list of texts.
    - evaluate(self, X_test, y_test): Evaluate the model on the test data.
    - preprocess_text(self, text): Preprocess a text (tokenize, remove stopwords, lemmatize).
    - plot_confusion_matrix(self, y_true, y_pred): Plot the confusion matrix.
    """

    def __init__(self, l2_reg=None):
        """
        Initialize the model with optional L2 regularization.
        
        :param l2_reg: L2 regularization factor (default is None)
        """
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        self.encoder = LabelEncoder()
        if l2_reg is not None:
            self.l2_reg = l2(l2_reg)
        else:
            self.l2_reg = None
        self.model = self.build_model()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def build_model(self):
        """
        Build a neural network model with optional L2 regularization.
        
        :return: Compiled neural network model
        """
        model = Sequential()
        model.add(Dense(512, input_shape=(10000,), activation='relu', kernel_regularizer=self.l2_reg))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu', kernel_regularizer=self.l2_reg))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):
        """
        Train the model on the training data and validate on the validation data.
        
        :param X_train: Training data (features)
        :param y_train: Training data (labels)
        :param X_val: Validation data (features)
        :param y_val: Validation data (labels)
        :param epochs: Number of epochs to train (default is 10)
        :param batch_size: Batch size for training (default is 128)
        """
        X_train_vectorized = self.vectorizer.fit_transform(X_train).todense()
        X_val_vectorized = self.vectorizer.transform(X_val).todense()
        y_train_encoded = to_categorical(self.encoder.fit_transform(y_train))
        y_val_encoded = to_categorical(self.encoder.transform(y_val))
        self.history = self.model.fit(X_train_vectorized, y_train_encoded, epochs=epochs, batch_size=batch_size,
                                      validation_data=(X_val_vectorized, y_val_encoded), verbose=1)

    def plot_accuracy(self):
        """
        Plot the training and validation accuracy over epochs.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['accuracy'], label='Train Accuracy', marker='o')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy', marker='o')
        plt.title('Model Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, model_path, vectorizer_path, encoder_path):
        """
        Save the trained model, vectorizer, and encoder to disk.
        
        :param model_path: Path to save the model
        :param vectorizer_path: Path to save the vectorizer
        :param encoder_path: Path to save the encoder
        """
        self.model.save(model_path)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.encoder, f)

    def load_model(self, model_path, vectorizer_path, encoder_path):
        """
        Load a trained model, vectorizer, and encoder from disk.
        
        :param model_path: Path to load the model from
        :param vectorizer_path: Path to load the vectorizer from
        :param encoder_path: Path to load the encoder from
        """
        self.model = load_model(model_path)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)

    def predict_texts(self, texts):
        """
        Predict the sentiment of a list of texts.
        
        :param texts: List of texts to predict
        :return: Predicted sentiment labels
        """
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        text_vectorized = self.vectorizer.transform(preprocessed_texts).todense()
        predictions = self.model.predict(text_vectorized)
        predicted_labels_indices = np.argmax(predictions, axis=1)
        predicted_labels = self.encoder.inverse_transform(predicted_labels_indices)
        return predicted_labels

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data.
        
        :param X_test: Test data (features)
        :param y_test: Test data (labels)
        :return: Accuracy of the model on the test data
        """
        X_test_vectorized = self.vectorizer.transform(X_test).todense()
        predictions = self.model.predict(X_test_vectorized)
        predicted_labels = np.argmax(predictions, axis=1)
        y_test_encoded = self.encoder.transform(y_test)
        accuracy = accuracy_score(y_test_encoded, predicted_labels) * 100
        self.plot_confusion_matrix(y_test_encoded, predicted_labels)
        return accuracy

    def preprocess_text(self, text):
        """
        Preprocess a text by tokenizing, removing stopwords, and lemmatizing.
        
        :param text: Text to preprocess
        :return: Preprocessed text
        """
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        processed_text = ' '.join(lemmatized_tokens)
        return processed_text

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot the confusion matrix for the predictions.
        
        :param y_true: True labels
        :param y_pred: Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cm, cmap=plt.cm.Blues)
        ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
        ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
        ax.set_ylim(1.5, -0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
        plt.title(f'Confusion Matrix - L2 Regularization: {self.l2_reg if self.l2_reg else "Not used"}')
        plt.show()

# Example usage:
# model_path = r'Sentiment_Model\sentiment_analysis_model_on_allData_balaced_dataset.h5'
# vectorizer_path = r'Sentiment_Model\vectorizer.pkl'
# encoder_path = r'Sentiment_Model\encoder.pkl'

# # Initialize model without training
# model = SentimentAnalysisModel(model_path, vectorizer_path, encoder_path, l2_reg=0.01)

# # Example of making a prediction. Try them if you wish!!!!!!!!!!!
# test_texts = [
#     "I absolutely love this new laptop from TechCorp! It has exceeded my expectations in terms of performance and battery life. I can use it all day without needing a charge, and it handles all my tasks effortlessly.",
#     "This movie was a complete waste of time. The plot was predictable, the acting was wooden, and I found myself checking my watch every few minutes. Definitely one of the worst movies I've seen this year.",
#     "The customer service at this store needs improvement. I was kept waiting for over an hour just to return a product, and the staff seemed disinterested and unhelpful throughout the process.",
#     "The meal was okay. I visited the new Italian restaurant in town and ordered their spaghetti. It wasn't bad, but it wasn't memorable either. Just your average pasta dish, nothing to write home about.",
#     "I recently purchased a blender from HomeTech. It’s fantastic! It blends everything to the perfect consistency and is very easy to clean. I’ve used it almost every day to make smoothies and soups.",
#     "This book is neither good nor bad. It's fairly mediocre. There are parts that I enjoyed, but overall it was just another run-of-the-mill science fiction novel that doesn't bring anything new to the genre.",
#     "I had an issue with my phone's battery, so I took it to the service center. The technicians there were knowledgeable and resolved the problem quickly. They even gave me tips on how to extend my battery life.",
#     "I'm disappointed with my purchase of the EasyVac vacuum cleaner. It fails to pick up finer particles of dust and is quite loud. I'm considering returning it and looking for a better option.",
#     "The conference was decent. The speakers were knowledgeable, but the topics covered were very basic and not as advanced as I had hoped. It would be nice if next time they could include some more in-depth discussions.",
#     "Our trip to Bali was breathtaking! The beaches were pristine, the food was exquisite, and the locals were incredibly welcoming. This trip was everything we dreamed of and more."
# ]
# positive_reviews = [
#     "I absolutely love the new ceramic knife set I purchased from Culinary Masters. The blades are incredibly sharp and have held their edge even after months of daily use. The ergonomic handles make slicing through even the toughest vegetables a breeze. I couldn't be happier with these knives and highly recommend them to any home chef.",
#     "The latest smartphone from TechGiant has truly revolutionized my daily routine. The face recognition feature works flawlessly, unlocking the phone instantly every time. Its processing speed is lightning-fast, allowing me to multitask without any lag. The large screen and crystal-clear resolution make it perfect for both work and leisure. Definitely a worthwhile investment!",
#     "I've been using the new FlexFit smart yoga mat for a few weeks now, and it's amazing how it has transformed my practice. The built-in sensors help ensure I am aligning my poses correctly, and the app integration provides real-time feedback and personalized session plans. It's like having a yoga instructor in my living room. Absolutely fantastic for anyone serious about yoga!",
#     "I recently tried the EcoBrew coffee maker, and I must say it's a game-changer for coffee enthusiasts. The ability to customize brew strength, temperature, and even grind size allows for a perfect cup every time. The sustainable filters and energy-efficient design also make it a guilt-free pleasure. Coffee has never tasted so good!",
#     "This new model of ergonomic office chair from SitRight has made my long hours at the desk so much more comfortable. The adjustable lumbar support and breathable mesh back have significantly reduced my back pain. The chair's smooth mobility and sturdy design also add to an overall excellent product that I would recommend to anyone spending a lot of time at a desk."
# ]
# print("Predicted sentiment:", model.predict_texts(test_texts))
