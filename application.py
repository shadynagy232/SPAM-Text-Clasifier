import os
import re
import string
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences


class messageClassification:
    def __init__(self):

        self.cd = os.getcwd()

    def proc(self, text):
        text = text.lower()
        text = re.sub(r"https?://\S+|www.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z]+", " ", text)
        text = re.sub(r"[0-9]", "", text)
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator)
        words = word_tokenize(text)
        words = [word for word in words if word not in stopwords.words("english")]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        text = " ".join(words)
        return text

    def test(self, message):

        loadedModel = load_model(self.cd + "\\Final Model.h5")
        testText = str(message)
        testText = self.proc(testText)
        testText = [testText]
        with open(self.cd + "\\tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        testText = tokenizer.texts_to_sequences(testText)
        testText = pad_sequences(testText, maxlen=442)
        threshold = 0.5
        result = loadedModel.predict(testText, verbose=2)
        result = result > threshold
        result = result.astype(int)
        if result[0] == 1:
            return "Spam"
        elif result[0] == 0:
            return "Ham"
